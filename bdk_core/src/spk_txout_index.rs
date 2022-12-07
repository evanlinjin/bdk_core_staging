use crate::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    ForEachTxout,
};
use bitcoin::{self, OutPoint, Script, Transaction, TxOut, Txid};

/// An index storing [`TxOut`]s that have a script pubkey that matches those in a list.
///
/// The basic idea is that you insert script pubkeys you care about into the index with [`add_spk`]
/// and then when you call [`scan`] the index will look at any txouts you pass in and
/// store and index any txouts matching one of its script pubkeys.
///
/// Each script pubkey is associated with a application defined index script index `I` which must be
/// [`Ord`]. Usually this is used to associate the derivation index of the script pubkey or even a
/// combination of `(keychain, derivation_index)`.
///
/// Note there is no harm in scanning transactions that disappear from the blockchain or were never
/// in there in the first place. `SpkTxOutIndex` is intentionally *monotone* -- you cannot delete or
/// modify txouts that have been indexed. To find out which txouts from the index are actually in the
/// chain or unspent etc you must use other sources of information like a [`SparseChain`].
///
/// [`TxOut`]: bitcoin::TxOut
/// [`add_spk`]: Self::add_spk
/// [`Ord`]: core::cmp::Ord
/// [`scan`]: Self::scan
/// [`SparseChain`]: crate::sparse_chain::SparseChain
#[derive(Clone, Debug)]
pub struct SpkTxOutIndex<I> {
    /// script pubkeys ordered by index
    script_pubkeys: BTreeMap<I, Script>,
    /// A reverse lookup from spk to spk index
    spk_indexes: HashMap<Script, I>,
    /// The set of unused indexes.
    unused: BTreeSet<I>,
    /// Lookup index and txout by outpoint.
    txouts: BTreeMap<OutPoint, (I, TxOut)>,
    /// Lookup from spk index to outpoints that had that spk
    spk_txouts: BTreeMap<I, HashSet<OutPoint>>,
}

impl<I> Default for SpkTxOutIndex<I> {
    fn default() -> Self {
        Self {
            txouts: Default::default(),
            script_pubkeys: Default::default(),
            spk_indexes: Default::default(),
            spk_txouts: Default::default(),
            unused: Default::default(),
        }
    }
}

impl<I: Clone + Ord> SpkTxOutIndex<I> {
    /// Scans an object containing many txouts.
    ///
    /// Typically this is used in two situations:
    ///
    /// 1. After loading transaction data from disk you may scan over all the txouts to restore all
    /// your txouts.
    /// 2. When getting new data from the chain you usually scan it before incorporating it into your chain state.
    ///
    /// See [`ForEachTxout`] for the types that support this.
    ///
    /// [`ForEachTxout`]: crate::ForEachTxout
    pub fn scan(&mut self, txouts: &impl ForEachTxout) {
        txouts.for_each_txout(&mut |(op, txout)| self.scan_txout(op, txout))
    }

    /// Scan a single `TxOut` for a matching script pubkey
    pub fn scan_txout(&mut self, op: OutPoint, txout: &TxOut) {
        if let Some(spk_i) = self.index_of_spk(&txout.script_pubkey) {
            self.txouts
                .insert(op.clone(), (spk_i.clone(), txout.clone()));
            self.spk_txouts
                .entry(spk_i.clone())
                .or_default()
                .insert(op.clone());
            self.unused.remove(&spk_i);
        }
    }

    /// Iterate over all known txouts that spend to tracked script pubkeys.
    pub fn iter_txout(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&I, OutPoint, &TxOut)> + ExactSizeIterator {
        self.txouts
            .iter()
            .map(|(op, (index, txout))| (index, *op, txout))
    }

    /// Finds all txouts on a transaction that has previously been scanned and indexed.
    pub fn txouts_in_tx(
        &self,
        txid: Txid,
    ) -> impl DoubleEndedIterator<Item = (&I, OutPoint, &TxOut)> {
        self.txouts
            .range(OutPoint::new(txid, u32::MIN)..=OutPoint::new(txid, u32::MAX))
            .map(|(op, (index, txout))| (index, *op, txout))
    }

    /// Returns the txout and script pubkey index of the `TxOut` at `OutPoint`.
    ///
    /// Returns `None` if the `TxOut` hasn't been scanned or if nothing matching was found there.
    pub fn txout(&self, outpoint: OutPoint) -> Option<(&I, &TxOut)> {
        self.txouts
            .get(&outpoint)
            .map(|(spk_i, txout)| (spk_i, txout))
    }

    /// Returns the script that has been inserted at the `index`.
    ///
    /// If that index hasn't been inserted yet it will return `None`.
    pub fn spk_at_index(&self, index: &I) -> Option<&Script> {
        self.script_pubkeys.get(index)
    }

    /// The script pubkeys being tracked by the index.
    pub fn script_pubkeys(&self) -> &BTreeMap<I, Script> {
        &self.script_pubkeys
    }

    /// Adds a script pubkey to scan for.
    ///
    /// the index will look for outputs spending to whenever it scans new data.
    pub fn add_spk(&mut self, index: I, spk: Script) {
        self.spk_indexes.insert(spk.clone(), index.clone());
        self.script_pubkeys.insert(index.clone(), spk);
        self.unused.insert(index);
    }

    /// Iterate over the script pubkeys that have been derived but do not have a transaction spending to them.
    pub fn iter_unused(
        &self,
    ) -> impl DoubleEndedIterator<Item = (&I, &Script)> + ExactSizeIterator {
        self.unused
            .iter()
            .map(|index| (index, self.spk_at_index(index).expect("must exist")))
    }

    /// Returns whether the script pubkey at index `index` has been used or not.
    ///
    /// i.e. has a transaction which spends to it.
    pub fn is_used(&self, index: &I) -> bool {
        self.spk_txouts
            .get(index)
            .map(|set| !set.is_empty())
            .unwrap_or(false)
    }

    /// Returns the index associated with the script pubkey.
    pub fn index_of_spk(&self, script: &Script) -> Option<I> {
        self.spk_indexes.get(script).cloned()
    }

    /// Whether any of the inputs of this transaction spend a txout tracked or whether any output
    /// matches one of our script pubkeys.
    ///
    /// It is easily possible to misuse this method and get false negatives by calling it before you
    /// have scanned the `TxOut`s the transaction is spending. For example if you want to filter out
    /// all the transactions in a block that are irrelevant you **must first scan all the
    /// transactions in the block** and only then use this method.
    pub fn is_relevant(&self, tx: &Transaction) -> bool {
        let input_matches = tx
            .input
            .iter()
            .find(|input| self.txouts.contains_key(&input.previous_output))
            .is_some();
        let output_matches = tx
            .output
            .iter()
            .find(|output| self.spk_indexes.contains_key(&output.script_pubkey))
            .is_some();
        input_matches || output_matches
    }
}