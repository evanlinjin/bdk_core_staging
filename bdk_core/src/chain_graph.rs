use crate::{
    sparse_chain::{self, ChainIndex, SparseChain},
    tx_graph::{TxGraph},
    BlockId, FullTxOut, TxHeight,
};
use alloc::vec::Vec;
use bitcoin::{OutPoint, Transaction, TxOut, Txid};
use core::fmt::Debug;

/// A convenient combination of a [`SparseChain<I>`] and a [`TxGraph`].
///
/// Very often you want to store transaction data when you record a transaction's existence. Adding
/// a transaction to a `ChainGraph` atomically stores the `txid` in its `SparseChain<I>`
/// while also storing the transaction data in its `TxGraph`.
///
/// The `ChainGraph` does not guarantee any 1:1 mapping between transactions in the `chain` and
/// `graph` or vis versa. Both fields are public so they can mutated indepdendly. Even if you only
/// modify the `ChainGraph` through its atomic API, keep in mind that `TxGraph` does not allow
/// deletions while `SparseChain` does so deleting a transaction from the chain cannot delete it
/// from the graph.
#[derive(Clone, Debug, PartialEq)]
pub struct ChainGraph<I = TxHeight> {
    chain: SparseChain<I>,
    graph: TxGraph,
}

impl<I> Default for ChainGraph<I> {
    fn default() -> Self {
        Self {
            chain: Default::default(),
            graph: Default::default(),
        }
    }
}

impl<I: ChainIndex> ChainGraph<I> {
    pub fn chain(&self) -> &SparseChain<I> {
        &self.chain
    }

    pub fn graph(&self) -> &TxGraph {
        &self.graph
    }

    pub fn mut_graph(&mut self) -> &mut TxGraph {
        &mut self.graph
    }

    /// Given a list of txids, returns a list of txids that are missing from the internal graph.
    pub fn determine_missing<'a>(
        &'a self,
        txids: impl Iterator<Item = Txid> + 'a,
    ) -> impl Iterator<Item = Txid> + 'a {
        txids.filter(|&txid| self.graph.tx(txid).is_none())
    }

    /// Inserts a transaction into the internal graph (and optionally the chain if `index` is
    /// [`Some`]). Returns a tuple of booleans: `(<is_chain_updated>, <is_graph_updated>)`.
    pub fn insert_tx(
        &mut self,
        tx: Transaction,
        index: Option<I>,
    ) -> Result<(bool, bool), sparse_chain::InsertTxErr> {
        let chain_updated = match index {
            Some(index) => self.chain.insert_tx(tx.txid(), index)?,
            None => false,
        };

        let graph_updated = self.graph.insert_tx(tx);

        Ok((chain_updated, graph_updated))
    }

    /// Inserts a list of transactions, and returns a tuple recording modification counts of the 
    /// internal chain and graph.
    pub fn insert_txs(
        &mut self,
        tx_iter: impl Iterator<Item = (Transaction, Option<I>)>,
    ) -> Result<(usize, usize), sparse_chain::InsertTxErr> {
        tx_iter.try_fold(
            (0, 0),
            |(mut chain_changes, mut graph_changes), (tx, index)| {
                let (chain_updated, graph_updated) = self.insert_tx(tx, index)?;
                if chain_updated {
                    chain_changes += 1;
                }
                if graph_updated {
                    graph_changes += 1;
                }
                Ok((chain_changes, graph_changes))
            },
        )
    }

    pub fn insert_output(&mut self, outpoint: OutPoint, txout: TxOut) -> bool {
        self.graph.insert_txout(outpoint, txout)
    }

    pub fn insert_checkpoint(
        &mut self,
        block_id: BlockId,
    ) -> Result<bool, sparse_chain::InsertCheckpointErr> {
        self.chain.insert_checkpoint(block_id)
    }

    /// Calculates the difference between self and `update` in the form of a [`ChangeSet`], while
    /// ensuring the histories of the update is consistent with the original chain.
    /// 
    /// It is assumed that the update chain is the most recent, and any transactions in the original
    /// chain that conflicts the update is evicted.
    /// 
    /// TODO: Would it make sense for the changeset to use the same type as a sparse_chain?
    pub fn determine_consistent_changeset(
        &self,
        update: &sparse_chain::SparseChain<I>,
    ) -> Result<sparse_chain::ChangeSet<I>, sparse_chain::UpdateFailure<I>> {
        let (mut changeset, invalid_from) = self.chain.determine_changeset(update)?;
        let invalid_from: TxHeight = invalid_from.into();

        let full_txs = update
            .iter_txids()
            // skip txids that already exist in the original chain (for efficiency)
            .filter(|&(_, txid)| self.chain.tx_index(*txid).is_none())
            .map(|&(_, txid)| self.graph.tx(txid))
            .collect::<Option<Vec<_>>>().ok_or(sparse_chain::UpdateFailure::MissingFullTxs)?;

        let conflicting_txids = full_txs
            .iter()
            .flat_map(|update_tx| {
                self.graph
                    .conflicting_txids(update_tx)
                    .filter_map(|(_, txid)| self.chain.tx_index(txid).map(|i| (txid, i)))
            });

        for (txid, original_index) in conflicting_txids {
            // if the evicted txid lies before "invalid_from", we screwed up
            if original_index.height() < invalid_from {
                return Err(sparse_chain::UpdateFailure::<I>::InconsistentTx {
                    inconsistent_txid: txid,
                    original_index: original_index.clone(),
                    update_index: None,
                });
            }

            changeset.txids.insert(txid, None);
        }

        Ok(changeset)
    }

    /// Applies a [`ChangeSet`] to the chain graph
    pub fn apply_changeset(&mut self, changeset: sparse_chain::ChangeSet<I>) -> Result<(), Vec<Txid>> {
        let missing = self.determine_missing(changeset.tx_additions()).collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(missing);
        }

        self.chain.apply_changeset(changeset);
        Ok(())
    }

    /// Applies the `update` chain graph. Note this is shorthand for calling [`determine_changeset`]
    /// and [`apply_changeset`] in sequence.
    ///
    /// [`apply_changeset`]: Self::apply_changeset
    /// [`determine_changeset`]: Self::determine_changeset
    pub fn apply_update(&mut self, update: sparse_chain::SparseChain<I>) -> Result<(), sparse_chain::UpdateFailure<I>> {
        let changeset = self.determine_consistent_changeset(&update)?;
        self.apply_changeset(changeset);
        Ok(())
    }

    /// Get the full transaction output at an outpoint if it exists in the chain and the graph.
    pub fn full_txout(&self, outpoint: OutPoint) -> Option<FullTxOut<I>> {
        self.chain.full_txout(&self.graph, outpoint)
    }

    /// Finds the transaction in the chain that spends `outpoint` given the input/output
    /// relationships in `graph`. Note that the transaction including `outpoint` does not need to be
    /// in the `graph` or the `chain` for this to return `Some(_)`.
    pub fn spent_by(&self, outpoint: OutPoint) -> Option<(&I, Txid)> {
        self.chain.spent_by(&self.graph, outpoint)
    }
}

impl<I> AsRef<TxGraph> for ChainGraph<I> {
    fn as_ref(&self) -> &TxGraph {
        &self.graph
    }
}
