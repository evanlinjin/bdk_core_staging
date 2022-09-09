use core::cmp::Ordering;

use crate::{collections::BTreeSet, Vec};
use bitcoin::{LockTime, Transaction, TxOut};

pub const TXIN_FIXED_WEIGHT: u32 = (32 + 4 + 4) * 4;

/// This is a common structure that is to be shared across all [`Selector`] instances.
/// This structure should not be mutated after forming.
#[derive(Debug, Clone)]
pub struct SelectorCommon {
    /// Selector options.
    pub opts: CoinSelectorOpt,

    /// Fixed inputs (if any).
    pub fixed_inputs: Option<InputGroup>,
    /// Input candidates.
    pub candidate_inputs: Vec<InputGroup>,

    /// Cost of creating change output + cost of spending the change output in the future.
    /// `change_weight * effective_feerate + spend_change_weight * long_term_feerate`
    pub cost_of_change: i64,

    /// Effective selection target (we take into consideration the fee required).
    /// `recipients_sum + fixed_weight * effective_feerate`
    pub effective_target: i64,
}

impl SelectorCommon {
    pub fn new(
        opts: CoinSelectorOpt,
        mut fixed_inputs: Option<InputGroup>,
        mut candidate_inputs: Vec<InputGroup>,
    ) -> Self {
        let cost_of_change = (opts.drain_weight as f32 * opts.effective_feerate
            + opts.drain_spend_weight as f32 * opts.long_term_feerate)
            .ceil() as i64;

        let (fixed_input_weight, fixed_input_value) =
            fixed_inputs.map(|i| (i.weight, i.value)).unwrap_or((0, 0));

        let fixed_weight = fixed_input_weight + opts.fixed_weight;
        let actual_target = opts.recipients_sum as i64 - fixed_input_value as i64;

        let effective_target =
            actual_target + (fixed_weight as f32 * opts.effective_feerate).ceil() as i64;

        // init all input candidates
        fixed_inputs
            .iter_mut()
            .chain(candidate_inputs.iter_mut())
            .for_each(|i| i.init(&opts));

        Self {
            opts,
            fixed_inputs,
            candidate_inputs,
            cost_of_change,
            effective_target,
        }
    }

    pub fn is_feerate_decreasing(&self) -> bool {
        self.opts.effective_feerate > self.opts.long_term_feerate
    }
}

#[derive(Debug, Clone)]
pub struct Selector {
    candidates: Vec<usize>,
    selected: BTreeSet<usize>,
    selected_state: InputGroup,
}

impl Selector {
    pub fn new(candidate_count: usize) -> Self {
        Self {
            candidates: (0..candidate_count).collect::<Vec<_>>(),
            selected: BTreeSet::new(),
            selected_state: InputGroup::empty(),
        }
    }

    pub fn new_sorted<F>(common: &SelectorCommon, mut sort: F) -> Self
    where
        F: FnMut(&(usize, &InputGroup), &(usize, &InputGroup)) -> Ordering,
    {
        let mut selector = Self::new(common.candidate_inputs.len());
        selector.candidates.sort_unstable_by(|&a, &b| {
            sort(
                &(a, &common.candidate_inputs[a]),
                &(b, &common.candidate_inputs[b]),
            )
        });
        selector
    }

    pub fn select(&mut self, common: &SelectorCommon, pos: usize) {
        assert!(pos < self.candidates.len());
        assert_eq!(self.candidates.len(), common.candidate_inputs.len());

        if self.selected.insert(pos) {
            let index = self.candidates[pos];
            self.selected_state
                .add_with(&common.candidate_inputs[index]);
        }
    }

    pub fn select_all(&mut self, common: &SelectorCommon) {
        (0..self.candidates.len()).for_each(|pos| self.select(common, pos))
    }

    pub fn deselect(&mut self, common: &SelectorCommon, pos: usize) {
        assert!(pos < self.candidates.len());
        assert_eq!(self.candidates.len(), common.candidate_inputs.len());

        if self.selected.remove(&pos) {
            let index = self.candidates[pos];
            self.selected_state
                .sub_with(&common.candidate_inputs[index]);
        }
    }

    pub fn last_selected(&self) -> Option<usize> {
        self.selected.iter().last().cloned()
    }

    pub fn is_selected(&self, pos: usize) -> bool {
        self.selected.contains(&pos)
    }

    pub fn iter_selected_positions(&self) -> impl Iterator<Item = usize> + '_ {
        self.selected.iter().cloned()
    }

    pub fn iter_unselected_positions(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.candidates.len()).filter(|pos| !self.selected.contains(pos))
    }

    pub fn iter_selected<'c>(&'c self, common: &'c SelectorCommon) -> impl Iterator<Item = (usize, &'c InputGroup)> + 'c {
        self.selected.iter().map(|pos| self.candidates[*pos]).map(|index| (index, &common.candidate_inputs[index]))
    }

    pub fn count(&self) -> usize {
        self.selected.len()
    }

    pub fn is_empty(&self) -> bool {
        self.selected.len() == 0
    }

    pub fn state(&self) -> &InputGroup {
        &self.selected_state
    }

    /// Excess is the difference between selected value and target
    /// `selected_effective_value - effective_target`
    pub fn excess(&self, common: &SelectorCommon) -> i64 {
        self.selected_state.effective_value - common.effective_target
    }

    /// Returns the current weight of the given selection.
    pub fn weight(&self, common: &SelectorCommon, include_drain: bool) -> u32 {
        let is_segwit = self.is_segwit(common);

        let extra_witness_weight = if is_segwit { 2_u32 } else { 0_u32 };
        let extra_varint_weight = self.extra_varint_weight(common);
        let fixed_inputs_weight = common.fixed_inputs.map(|i| i.weight).unwrap_or(0);
        let drain_weight = if include_drain {
            common.opts.drain_weight
        } else {
            0
        };

        common.opts.fixed_weight // fixed outputs and header fields
            + fixed_inputs_weight // fixed inputs
            + self.selected_state.weight // selected inputs
            + drain_weight // drain output(s) (if any)
            + extra_witness_weight // extra witness headers (if any)
            + extra_varint_weight // extra input len varint weight (if any)
    }

    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    pub fn candidate<'c>(&self, common: &'c SelectorCommon, pos: usize) -> &'c InputGroup {
        assert!(pos < self.candidates.len());
        assert_eq!(self.candidates.len(), common.candidate_inputs.len());

        let index = self.candidates[pos];
        &common.candidate_inputs[index]
    }

    /// Whether the current selection (inclusive of fixed inputs) contain at least one segwit input.
    pub fn is_segwit(&self, common: &SelectorCommon) -> bool {
        let fixed_segwit_count = common.fixed_inputs.map(|i| i.segwit_count).unwrap_or(0);
        self.selected_state.segwit_count + fixed_segwit_count > 0
    }

    /// This is the extra weight of the `txin_count` variable (which is a `varint`), when we
    /// introduce all inputs.
    pub fn extra_varint_weight(&self, common: &SelectorCommon) -> u32 {
        let input_count = self.selected_state.input_count
            + common.fixed_inputs.map(|i| i.input_count).unwrap_or(0);

        varint_size(input_count).saturating_sub(1) * 4
    }

    pub fn finish(
        &self,
        common: &SelectorCommon,
        min_viable_change: i64, // for BnB: Cost of change
    ) -> Result<Selection, SelectionFailure> {
        let fixed_inputs = &common.fixed_inputs.unwrap_or_else(InputGroup::empty);
        let selected_inputs = self.state();

        let excess = self.excess(common);
        if excess < 0 {
            return Err(SelectionFailure::InsufficientFunds {
                selected: selected_inputs.effective_value as _,
                needed: common.effective_target as _,
            });
        }
        assert!(selected_inputs.value + fixed_inputs.value > common.opts.recipients_sum);

        // fee of creating change output(s) (not to be mistaken as cost_of_change)
        let change_fee =
            (common.opts.drain_weight as f32 * common.opts.effective_feerate).ceil() as i64;

        // find the value that should be used for our change output (if any)
        let change_value = {
            let change_value = excess - change_fee;
            if change_value < min_viable_change {
                None
            } else {
                Some(change_value)
            }
        };

        let use_drain = change_value.is_some();
        let waste = fixed_inputs.waste
            + selected_inputs.waste
            + if use_drain {
                common.opts.drain_cost() as i64
            } else {
                excess
            };
        let total_weight = self.weight(common, use_drain);
        let fee = fixed_inputs.value + selected_inputs.value
            - common.opts.recipients_sum
            - change_value.unwrap_or(0) as u64;
        let excess = excess - fee as i64;

        Ok(Selection {
            selected: self
                .selected
                .iter()
                .map(|&pos| self.candidates[pos])
                .collect(),
            excess: excess as _,
            fee,
            use_drain,
            total_weight,
            waste,
        })
    }
}

#[derive(Debug, Clone)]
pub struct CoinSelector<'a> {
    opts: &'a CoinSelectorOpt,
    candidates: &'a Vec<InputGroup>,

    /* The following fields record the selection state */
    selected_indexes: BTreeSet<usize>, // indexes of selected input candidates
    selected_state: SelectedState,     // state of the selected inputs
}

/// Represents the state of the selected input candidates.
#[derive(Debug, Clone)]
pub struct SelectedState {
    pub waste: i64,                     // this is the waste of selected inputs only
    pub value: u64,                     // sum of selected input values
    pub value_remaining: u64,           // remaining unselected input values
    pub effective_value: i64,           // sum of selected effective values
    pub effective_value_remaining: i64, // remaining unselected effective values
    pub input_count: usize,             // accumulated count of all inputs
    pub segwit_count: usize,            // number of segwit inputs
    pub weight: u32,                    // accumulated weight of all selected inputs
}

impl SelectedState {
    pub fn add_candidate(&mut self, candidate: &InputGroup) {
        self.waste += candidate.waste;
        self.value += candidate.value;
        self.value_remaining -= candidate.value;
        self.effective_value += candidate.effective_value;
        self.effective_value_remaining -= candidate.effective_value;
        self.input_count += candidate.input_count;
        self.segwit_count += candidate.segwit_count;
        self.weight += candidate.weight;
    }

    pub fn sub_candidate(&mut self, candidate: &InputGroup) {
        self.waste -= candidate.waste;
        self.value -= candidate.value;
        self.value_remaining += candidate.value;
        self.effective_value -= candidate.effective_value;
        self.effective_value_remaining += candidate.effective_value;
        self.input_count -= candidate.input_count;
        self.segwit_count -= candidate.segwit_count;
        self.weight -= candidate.weight;
    }

    pub fn is_segwit(&self) -> bool {
        self.segwit_count > 0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct InputGroup {
    /// Number of inputs contained within this [`InputCandidate`].
    /// If we are using single UTXOs as candidates, this would be 1.
    /// If we are working in `OutputGroup`s (as done in Bitcoin Core), this would be > 1.
    pub input_count: usize,
    /// Whether at least one input of this [`InputCandidate`] is spending a segwit output.
    pub segwit_count: usize,
    /// Total value of these input(s).
    pub value: u64,
    /// Weight of these input(s): `prevout + nSequence + scriptSig + scriptWitness` per input.
    pub weight: u32,

    /// This is the input(s) value minus cost of spending these input(s):
    /// `value - (weight * effective_fee)`
    effective_value: i64,
    /// This is the `waste` of including these input(s).
    /// `weight * (effective_fee - long_term_fee)`
    waste: i64,
}

impl InputGroup {
    /// New [`InputCandidate`].
    pub fn new_group(input_count: usize, value: u64, weight: u32, is_segwit: bool) -> Self {
        assert!(
            input_count > 0,
            "InputCandidate does not make sense with 0 inputs"
        );

        Self {
            input_count,
            value,
            weight,
            segwit_count: if is_segwit { 1 } else { 0 },

            // These values are set by `init`
            effective_value: 0,
            waste: 0,
        }
    }

    pub fn new_single(value: u64, weight: u32, is_segwit: bool) -> Self {
        Self::new_group(1, value, weight, is_segwit)
    }

    pub fn empty() -> Self {
        Self {
            input_count: 0,
            segwit_count: 0,
            value: 0,
            weight: 0,
            effective_value: 0,
            waste: 0,
        }
    }

    pub fn init(&mut self, opts: &CoinSelectorOpt) {
        self.effective_value =
            self.value as i64 - (self.weight as f32 * opts.effective_feerate).ceil() as i64;
        self.waste =
            (self.weight as f32 * (opts.effective_feerate - opts.long_term_feerate)).ceil() as i64;
    }

    pub fn add_with(&mut self, other: &Self) -> &Self {
        self.input_count += other.input_count;
        self.segwit_count += other.segwit_count;
        self.value += other.value;
        self.weight += other.weight;
        self.effective_value += other.effective_value;
        self.waste += other.waste;
        self
    }

    pub fn sub_with(&mut self, other: &Self) -> &Self {
        self.input_count -= other.input_count;
        self.segwit_count -= other.segwit_count;
        self.value -= other.value;
        self.weight -= other.weight;
        self.effective_value -= other.effective_value;
        self.waste -= other.waste;
        self
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

    /// The fixed weight of the template transaction, inclusive of: `nVersion`, `nLockTime`
    /// fixed `vout`s and the first bytes of `vin_len` and `vout_len`.
    ///
    /// Weight of the drain output is not included.
    pub fixed_weight: u32,
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

    /// This is the extra weight of the `txin_count` variable (which is a `varint`), when we
    /// introduce inputs on top of the "fixed" input count.
    pub fn extra_varint_weight(&self, total_input_count: usize) -> u32 {
        (varint_size(total_input_count) - 1) * 4
    }

    /// Selection target should be `recipients_sum + fixed_weight * effective_feerate`
    pub fn target_effective_value(&self) -> i64 {
        self.recipients_sum as i64
            + (self.fixed_weight as f32 * self.effective_feerate).ceil() as i64
    }
}

impl<'a> CoinSelector<'a> {
    pub fn new(candidates: &'a Vec<InputGroup>, opts: &'a CoinSelectorOpt) -> Self {
        let (unselected_value, unselected_effective_value) = candidates
            .iter()
            .map(|i| (i.value, i.effective_value))
            .fold((0, 0), |a, v| (a.0 + v.0, a.1 + v.1));

        Self {
            opts,
            candidates,

            selected_indexes: Default::default(),
            selected_state: SelectedState {
                waste: 0,
                value: 0,
                effective_value: 0,
                value_remaining: unselected_value,
                effective_value_remaining: unselected_effective_value,
                input_count: 0,
                segwit_count: 0,
                weight: 0,
            },
        }
    }

    pub fn candidates(&self) -> &[InputGroup] {
        &self.candidates
    }

    pub fn candidate(&self, index: usize) -> &InputGroup {
        assert!(index < self.candidates.len());
        &self.candidates[index]
    }

    pub fn options(&self) -> &CoinSelectorOpt {
        self.opts
    }

    pub fn select(&mut self, index: usize) {
        assert!(index < self.candidates.len());
        if self.selected_indexes.insert(index) {
            self.selected_state.add_candidate(&self.candidates[index]);
        }
    }

    pub fn deselect(&mut self, index: usize) {
        assert!(index < self.candidates.len());
        if self.selected_indexes.remove(&index) {
            self.selected_state.sub_candidate(&self.candidates[index]);
        }
    }

    /// Returns the current state of all inputs in the current selection.
    pub fn state(&self) -> &SelectedState {
        &self.selected_state
    }

    pub fn excess(&self) -> i64 {
        self.selected_state.effective_value - self.opts.target_effective_value()
    }

    pub fn current_weight_without_drain(&self) -> u32 {
        let inputs = self.state();
        let is_segwit = inputs.segwit_count > 0;

        let extra_witness_weight = if is_segwit { 2_u32 } else { 0_u32 };
        let extra_varint_weight = self.opts.extra_varint_weight(inputs.input_count);

        self.opts.fixed_weight + inputs.weight + extra_witness_weight + extra_varint_weight
    }

    pub fn iter_selected(&self) -> impl Iterator<Item = (usize, InputGroup)> + '_ {
        self.selected_indexes
            .iter()
            .map(|&index| (index, self.candidates[index]))
    }

    pub fn iter_unselected(&self) -> impl Iterator<Item = (usize, InputGroup)> + '_ {
        self.candidates
            .iter()
            .enumerate()
            .filter(|(index, _)| !self.selected_indexes.contains(index))
            .map(|(index, c)| (index, *c))
    }

    pub fn all_selected(&self) -> bool {
        self.selected_indexes.len() == self.candidates.len()
    }

    pub fn select_all(&mut self) {
        let unselected = self.iter_unselected().map(|(i, _)| i).collect::<Vec<_>>();
        for index in unselected {
            self.select(index);
        }
    }

    pub fn select_until_finished(&mut self) -> Result<Selection, SelectionFailure> {
        let mut selection = self.finish();

        if selection.is_ok() {
            return selection;
        }

        let unselected = self.iter_unselected().map(|(i, _)| i).collect::<Vec<_>>();
        for index in unselected {
            self.select(index);
            selection = self.finish();

            if selection.is_ok() {
                break;
            }
        }

        selection
    }

    pub fn finish(&self) -> Result<Selection, SelectionFailure> {
        let selected = self.state();

        // this is the tx weight without drain.
        let base_weight = {
            let is_segwit = selected.segwit_count > 0;
            let extra_witness_weight = if is_segwit { 2_u32 } else { 0_u32 };
            let extra_varint_weight = self.opts.extra_varint_weight(selected.input_count);
            self.opts.fixed_weight + selected.weight + extra_witness_weight + extra_varint_weight
        };

        if selected.value < self.opts.recipients_sum {
            return Err(SelectionFailure::InsufficientFunds {
                selected: selected.value,
                needed: self.opts.recipients_sum,
            });
        }

        let inputs_minus_outputs = selected.value - self.opts.recipients_sum;

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
                let implied_output_value = selected.value - target_fee_without_drain;
                match implied_output_value.checked_sub(self.opts.recipients_sum) {
                    Some(excess) => (excess, false),
                    None => {
                        return Err(SelectionFailure::InsufficientFunds {
                            selected: selected.value,
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
        let waste = selected.waste
            + if use_drain {
                self.opts.drain_cost()
            } else {
                excess
            } as i64;

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
    NoSolution,
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
            Self::NoSolution => {
                write!(f, "algorithm cannot find a solution")
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
const MAX_MONEY: i64 = 2_100_000_000_000_000;
const MAX_TRIES_BNB: usize = 100_000;

pub fn select_coins_bnb(current: &mut CoinSelector) -> Result<Selection, SelectionFailure> {
    let target_value = current.options().target_effective_value();
    let cost_of_change = current.options().drain_cost() as i64;

    // ensure we have enough to select with
    // TODO: Figure out how to enfore effective value to not be negative.
    {
        let state = current.state();
        let avaliable = state.effective_value + state.effective_value_remaining;
        if avaliable < target_value {
            // TODO: This error should use `i64`.
            return Err(SelectionFailure::InsufficientFunds {
                selected: avaliable as u64,
                needed: target_value as u64,
            });
        }
    }

    // remaining value we have left in the current branch
    let mut remaining_value = current.state().effective_value_remaining;

    // our best solution (start with the worst possible solution)
    let mut best = Option::<CoinSelector>::None; // current.clone();

    // sort unselected index pool in descending order in terms of effective value
    let mut pool_index = 0_usize;
    let mut selected_pool_indexes = Vec::<usize>::with_capacity(current.candidates().len());
    let pool = {
        let mut pool = current.iter_unselected().collect::<Vec<_>>();
        pool.sort_by(|(_, ca), (_, cb)| cb.effective_value.cmp(&ca.effective_value));
        pool
    };

    // depth-first loop
    for try_index in 0..MAX_TRIES_BNB {
        if try_index > 0 {
            pool_index += 1;
        }

        // conditions for starting a backtrack
        let backtrack = {
            let feerate_decreasing =
                current.options().effective_feerate > current.options().long_term_feerate;

            let current_value = current.state().effective_value;
            let current_waste = current.state().waste + current.excess();
            let best_waste = best
                .as_ref()
                .map(|b| b.state().waste + b.excess())
                .unwrap_or(MAX_MONEY);

            // TODO: Add comments
            if current_value + remaining_value < target_value
                || current_value > target_value + cost_of_change
                || (current.state().waste > best_waste && feerate_decreasing)
            {
                true
            } else if current_value >= target_value {
                if current_waste <= best_waste {
                    best.replace(current.clone());
                }

                true
            } else {
                false
            }
        };

        if backtrack {
            if current.state().value == 0 {
                break;
            }

            pool_index -= 1;
            while pool_index > *selected_pool_indexes.last().unwrap() {
                remaining_value += pool[pool_index].1.effective_value;
                pool_index -= 1;
            }
            assert_eq!(pool_index, *selected_pool_indexes.last().unwrap());

            let candidate_index = pool[pool_index].0;
            current.deselect(candidate_index);
            selected_pool_indexes.pop();

            continue;
        }

        let (candidate_index, candidate) = pool[pool_index];
        remaining_value -= candidate.effective_value;

        if current.state().value == 0
            || pool_index - 1 == *selected_pool_indexes.last().unwrap()
            || candidate.effective_value != pool[pool_index - 1].1.effective_value
            || candidate.weight != pool[pool_index - 1].1.weight
        {
            current.select(candidate_index);
            selected_pool_indexes.push(pool_index);
        }
    }

    let selection = best
        .as_ref()
        .ok_or(SelectionFailure::NoSolution)?
        .finish()?;

    assert_eq!(
        selection.waste,
        {
            let best = best.as_ref().unwrap();
            best.state().waste + best.excess()
        },
        "waste does not match up"
    );

    Ok(selection)
}

pub fn select_coins_bnb2(common: &SelectorCommon) -> Result<Selection, SelectionFailure> {
    let feerate_decreasing = common.is_feerate_decreasing();
    let target_value = common.effective_target;
    let change_cost = common.cost_of_change;

    // remaining value of the current branch
    let mut remaining_value = common
        .candidate_inputs
        .iter()
        .map(|i| i.effective_value)
        .sum::<i64>();

    // ensure we have enough to select with
    if remaining_value < target_value {
        return Err(SelectionFailure::InsufficientFunds {
            selected: remaining_value as _,
            needed: target_value as _,
        });
    }

    // current position
    let mut pos = 0_usize;

    // current selection
    // sort candidates by effective value (descending)
    // TODO: Should we filter candidates with negative effective values?
    let mut current = Selector::new_sorted(common, |&(_, c1), &(_, c2)| {
        c2.effective_value.cmp(&c1.effective_value)
    });

    // best selection
    let mut best = Option::<Selector>::None;

    // depth-fist loop
    for try_index in 0..MAX_TRIES_BNB {
        if try_index > 0 {
            pos += 1;
        }

        // conditions for a backtrack
        let backtrack = {
            let current_value = current.state().effective_value;
            let current_waste = current.state().waste;
            let best_waste = best
                .as_ref()
                .map(|b| b.state().waste + b.excess(common))
                .unwrap_or(MAX_MONEY);

            if current_value + remaining_value < target_value
                || current_value > target_value + change_cost
                || (current.state().waste > best_waste && feerate_decreasing)
            {
                true
            } else if current_value >= target_value {
                if current_waste <= best_waste {
                    best.replace(current.clone());
                }
                true
            } else {
                false
            }
        };

        if backtrack {
            let last_pos = match current.last_selected() {
                Some(last_pos) => last_pos,
                None => break, // nothing selected, all solutions search
            };

            (pos - 1..last_pos)
                .for_each(|pos| remaining_value += current.candidate(common, pos).effective_value);

            current.deselect(common, last_pos);
            pos = last_pos;
        } else {
            // continue down this branch
            let candidate = current.candidate(common, pos);

            // remove from remaining_value in branch
            remaining_value -= candidate.effective_value;

            // whether the previous position is selected
            let prev_pos_selected = current
                .last_selected()
                .map(|last_selected_pos| last_selected_pos == pos - 1)
                .unwrap_or(false);

            if current.is_empty()
                || prev_pos_selected
                || candidate.effective_value != current.candidate(common, pos - 1).effective_value
                || candidate.weight != current.candidate(common, pos - 1).weight
            {
                current.select(common, pos);
            }
        }
    }

    let selection = best
        .as_ref()
        .ok_or(SelectionFailure::NoSolution)?
        .finish(common, change_cost)?;

    assert_eq!(
        selection.waste,
        {
            let best = best.as_ref().unwrap();
            best.state().waste + best.excess(common)
        },
        "waste does not match up"
    );

    Ok(selection)
}
