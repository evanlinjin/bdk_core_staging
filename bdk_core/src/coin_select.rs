use crate::{collections::BTreeSet, Vec};
use bitcoin::{LockTime, Transaction, TxOut};

pub const TXIN_FIXED_WEIGHT: u32 = (32 + 4 + 4) * 4;

#[derive(Debug, Clone)]
pub struct CoinSelector {
    opts: CoinSelectorOpt,

    candidates: Vec<InputCandidate>,
    /// Indexes of selected candidates.
    selected_indexes: BTreeSet<usize>,

    /// Selection target value
    target_value: i64,
    current_selected_value: i64,
    current_unselected_value: i64,
}

#[derive(Debug, Clone, Copy)]
pub struct InputCandidate {
    /// Number of inputs contained within this [`InputCandidate`].
    /// If we are using single UTXOs as candidates, this would be 1.
    /// If we are working in `OutputGroup`s (as done in Bitcoin Core), this would be > 1.
    pub input_count: usize,
    /// Total value of these input(s).
    pub value: u64,
    /// This is the input(s) value minus cost of spending these input(s):
    /// `value - (weight * effective_fee)`
    pub effective_value: i64,
    /// Weight of these input(s): `prevout + nSequence + scriptSig + scriptWitness` per input.
    pub weight: u32,
    /// Whether at least one input of this [`InputCandidate`] is spending a segwit output.
    pub is_segwit: bool,
}

#[cfg(feature = "std")]
impl std::ops::Add for InputCandidate {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            input_count: self.input_count + other.input_count,
            value: self.value + other.value,
            effective_value: self.effective_value + other.effective_value,
            weight: self.weight + other.weight,
            is_segwit: self.is_segwit || other.is_segwit,
        }
    }
}

impl InputCandidate {
    /// New [`InputCandidate`] where `effective_value` is calculated from fee defined in `opts`.
    pub fn new(
        opts: &CoinSelectorOpt,
        input_count: usize,
        value: u64,
        weight: u32,
        is_segwit: bool,
    ) -> Self {
        assert!(
            input_count > 0,
            "InputCandidate does not make sense with 0 inputs"
        );
        let effective_value = value as i64 - (weight as f32 * opts.effective_feerate).ceil() as i64;
        Self {
            input_count,
            value,
            effective_value,
            weight,
            is_segwit,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CoinSelectorOpt {
    /// The sum of recipient output values (in satoshis).
    pub recipients_sum: u64,

    /// The feerate we should try and achieve in sats per weight unit.
    pub effective_feerate: f32,
    /// The long term feerate if we are to spend an input in the future instead (in sats/wu).
    /// This is used for calculating waste.
    pub long_term_feerate: f32,
    /// The minimum absolute fee (in satoshis).
    pub min_absolute_fee: u64,

    /// Additional weight if we use the drain (change) output(s).
    pub drain_weight: u32,
    /// Weight of a `txin` used to spend the drain output(s) later on.
    pub drain_spend_weight: u32,

    /// The additional fixed weight of the template transaction including: `nVersion`, `nLockTime`
    /// and fixed `vout`s and the first bytes of `vin_len` and `vout_len`. Weight of `vin` is not
    /// included. Drain output is not included.
    pub fixed_weight: u32,
    /// The additional fixed input(s) of the template transaction (if any).
    pub additional_fixed_input: Option<InputCandidate>,
}

impl CoinSelectorOpt {
    pub fn from_weights(fixed_weight: u32, drain_weight: u32, drain_spend_weight: u32) -> Self {
        Self {
            recipients_sum: 0,
            // 0.25 per wu i.e. 1 sat per byte
            effective_feerate: 0.25,
            long_term_feerate: 0.25,
            min_absolute_fee: 0,
            drain_weight,
            drain_spend_weight,
            fixed_weight,
            additional_fixed_input: None,
        }
    }

    pub fn fund_outputs(
        txouts: &[TxOut],
        drain_outputs: &[TxOut],
        drain_spend_weight: u32,
    ) -> Self {
        let mut tx = Transaction {
            input: vec![],
            version: 1,
            lock_time: LockTime::ZERO.into(),
            output: txouts.to_vec(),
        };
        let fixed_weight = tx.weight();

        // this awkward calculation is necessary since TxOut doesn't have \.weight()
        let drain_weight = {
            drain_outputs
                .iter()
                .for_each(|txo| tx.output.push(txo.clone()));
            tx.weight() - fixed_weight
        };

        Self {
            recipients_sum: txouts.iter().map(|txout| txout.value).sum(),
            ..Self::from_weights(fixed_weight as u32, drain_weight as u32, drain_spend_weight)
        }
    }

    /// Calculates the "cost of change": cost of creating drain output + cost of spending the drain
    /// output in the future.
    pub fn drain_cost(&self) -> u64 {
        ((self.effective_feerate * self.drain_weight as f32).ceil()
            + (self.long_term_feerate * self.drain_spend_weight as f32).ceil()) as u64
    }

    /// Fixed weight of the transaction, inclusive of fixed inputs and outputs, but exclusive of
    /// the drain output.
    pub fn fixed_weight(&self) -> u32 {
        self.additional_fixed_input
            .map(|i| i.weight)
            .unwrap_or(0_u32)
            + self.fixed_weight
    }

    /// This is the extra weight of the `txin_count` variable (which is a `varint`), when we
    /// introduce inputs on top of the "fixed" input count.
    pub fn extra_varint_weight(&self, total_input_count: usize) -> u32 {
        let fixed_count = self
            .additional_fixed_input
            .map(|i| i.input_count)
            .unwrap_or(0);

        let fixed_varint_size = varint_size(fixed_count);
        let total_varint_size = varint_size(total_input_count);

        (total_varint_size - fixed_varint_size) * 4
    }

    /// Selection target should be `recipients_sum - fixed_input_effective_value + fixed_weight * effective_feerate`
    pub fn selection_target(&self) -> i64 {
        self.recipients_sum as i64
            - self
                .additional_fixed_input
                .map(|i| i.effective_value)
                .unwrap_or(0)
            + (self.fixed_weight() as f32 * self.effective_feerate).ceil() as i64
    }
}

impl CoinSelector {
    pub fn new(candidates: Vec<InputCandidate>, opts: CoinSelectorOpt) -> Self {
        Self {
            candidates: candidates.clone(),
            selected_indexes: Default::default(),
            opts,

            target_value: opts.selection_target(),
            current_selected_value: 0,
            current_unselected_value: candidates.iter().map(|i| i.effective_value).sum(),
        }
    }

    pub fn candidates(&self) -> &[InputCandidate] {
        &self.candidates
    }

    pub fn select(&mut self, index: usize) {
        assert!(index < self.candidates.len());
        self.selected_indexes.insert(index);
    }

    /// Returns the current state of all inputs (both selected and fixed) in format
    /// `(input_count, input_total_weight, has_segwit)`.
    pub fn current_input_state(&self) -> (usize, u32, bool) {
        let (input_count, weight, is_segwit) = self
            .selected_indexes
            .iter()
            .map(|&i| &self.candidates[i])
            .chain(self.opts.additional_fixed_input.iter())
            .fold(
                (0_usize, 0_u32, false),
                |(input_count, weight, is_segwit), c| {
                    (
                        input_count + c.input_count,
                        weight + c.weight,
                        is_segwit || c.is_segwit,
                    )
                },
            );

        (input_count, weight, is_segwit)
    }

    pub fn current_weight_without_drain(&self) -> u32 {
        let (input_count, input_weight, is_segwit) = self.current_input_state();

        let extra_witness_weight = if is_segwit { 2_u32 } else { 0_u32 };
        let extra_varint_weight = self.opts.extra_varint_weight(input_count);

        self.opts.fixed_weight + input_weight + extra_witness_weight + extra_varint_weight
    }

    pub fn selected(&self) -> impl Iterator<Item = (usize, &InputCandidate)> + '_ {
        self.selected_indexes
            .iter()
            .map(|&index| (index, self.candidates.get(index).unwrap()))
    }

    pub fn unselected(&self) -> Vec<usize> {
        let all_indexes = (0..self.candidates.len()).collect::<BTreeSet<_>>();
        all_indexes
            .difference(&self.selected_indexes)
            .cloned()
            .collect()
    }

    pub fn all_selected(&self) -> bool {
        self.selected_indexes.len() == self.candidates.len()
    }

    pub fn select_all(&mut self) {
        for next_unselected in self.unselected() {
            self.select(next_unselected)
        }
    }

    pub fn select_until_finished(&mut self) -> Result<Selection, SelectionFailure> {
        let mut selection = self.finish();

        if selection.is_ok() {
            return selection;
        }

        for next_unselected in self.unselected() {
            self.select(next_unselected);
            selection = self.finish();

            if selection.is_ok() {
                break;
            }
        }

        selection
    }

    pub fn current_value(&self) -> u64 {
        self.opts
            .additional_fixed_input
            .map(|i| i.value)
            .unwrap_or(0_u64)
            + self.selected().map(|(_, wv)| wv.value).sum::<u64>()
    }

    pub fn finish(&self) -> Result<Selection, SelectionFailure> {
        let (input_count, input_weight, is_segwit) = self.current_input_state();

        // this is the tx weight without drain.
        let base_weight = {
            let extra_witness_weight = if is_segwit { 2_u32 } else { 0_u32 };
            let extra_varint_weight = self.opts.extra_varint_weight(input_count);
            self.opts.fixed_weight + input_weight + extra_witness_weight + extra_varint_weight
        };

        if self.current_value() < self.opts.recipients_sum {
            return Err(SelectionFailure::InsufficientFunds {
                selected: self.current_value(),
                needed: self.opts.recipients_sum,
            });
        }

        let inputs_minus_outputs = self.current_value() - self.opts.recipients_sum;

        // check fee rate satisfied
        let feerate_without_drain = inputs_minus_outputs as f32 / base_weight as f32;

        // we simply don't have enough fee to achieve the feerate
        if feerate_without_drain < self.opts.effective_feerate {
            return Err(SelectionFailure::FeerateTooLow {
                needed: self.opts.effective_feerate,
                had: feerate_without_drain,
            });
        }

        if inputs_minus_outputs < self.opts.min_absolute_fee {
            return Err(SelectionFailure::AbsoluteFeeTooLow {
                needed: self.opts.min_absolute_fee,
                had: inputs_minus_outputs,
            });
        }

        let weight_with_drain = base_weight + self.opts.drain_weight;

        let target_fee_with_drain =
            ((self.opts.effective_feerate * weight_with_drain as f32).ceil() as u64)
                .max(self.opts.min_absolute_fee);
        let target_fee_without_drain = ((self.opts.effective_feerate * base_weight as f32).ceil()
            as u64)
            .max(self.opts.min_absolute_fee);

        let (excess, use_drain) = match inputs_minus_outputs.checked_sub(target_fee_with_drain) {
            Some(excess) => (excess, true),
            None => {
                let implied_output_value = self.current_value() - target_fee_without_drain;
                match implied_output_value.checked_sub(self.opts.recipients_sum) {
                    Some(excess) => (excess, false),
                    None => {
                        return Err(SelectionFailure::InsufficientFunds {
                            selected: self.current_value(),
                            needed: target_fee_without_drain + self.opts.recipients_sum,
                        })
                    }
                }
            }
        };

        let (total_weight, fee) = if use_drain {
            (weight_with_drain, target_fee_with_drain)
        } else {
            (base_weight, target_fee_without_drain)
        };

        // `waste` is the waste of spending the inputs now (with the current selection), as opposed
        // to spending it later.
        let waste = {
            let base_waste = if use_drain {
                self.opts.drain_cost()
            } else {
                excess
            };
            base_waste as i64
                + input_weight as i64
                    * (self.opts.effective_feerate - self.opts.long_term_feerate) as i64
        };

        Ok(Selection {
            selected: self.selected_indexes.clone(),
            excess,
            use_drain,
            total_weight,
            fee,
            waste,
        })
    }
}

#[derive(Clone, Debug)]
pub enum SelectionFailure {
    InsufficientFunds { selected: u64, needed: u64 },
    FeerateTooLow { needed: f32, had: f32 },
    AbsoluteFeeTooLow { needed: u64, had: u64 },
}

impl core::fmt::Display for SelectionFailure {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SelectionFailure::InsufficientFunds { selected, needed } => write!(
                f,
                "insufficient coins selected, had {} needed {}",
                selected, needed
            ),
            SelectionFailure::FeerateTooLow { needed, had } => {
                write!(f, "feerate too low, needed {}, had {}", needed, had)
            }
            SelectionFailure::AbsoluteFeeTooLow { needed, had } => {
                write!(f, "absolute fee too low, needed {}, had {}", needed, had)
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SelectionFailure {}

#[derive(Clone, Debug)]
pub struct Selection {
    pub selected: BTreeSet<usize>,
    pub excess: u64,
    pub fee: u64,
    pub use_drain: bool,
    pub total_weight: u32,
    pub waste: i64,
}

impl Selection {
    pub fn apply_selection<'a, T>(
        &'a self,
        candidates: &'a [T],
    ) -> impl Iterator<Item = &'a T> + 'a {
        self.selected.iter().map(|i| &candidates[*i])
    }
}

/* HELPERS */

fn varint_size(v: usize) -> u32 {
    if v <= 0xfc {
        return 1;
    }
    if v <= 0xffff {
        return 3;
    }
    if v <= 0xffff_ffff {
        return 5;
    }
    return 9;
}

/* ALGORITHMS */

pub fn select_coins_bnb(
    opts: CoinSelectorOpt,
    candidates: Vec<InputCandidate>,
) -> Result<Selection, SelectionFailure> {
    let target_value = opts.selection_target();

    // ensure we have enough to select with
    candidates.iter().map(|i| i.value).sum::<u64>();
    todo!()
}
