use bitcoin::{OutPoint, Transaction, TxOut, Txid};
use core::fmt::Debug;

use crate::{
    BlockId, ChainIndexExtension, ChangeSet, InsertCheckpointErr, InsertTxErr, SparseChain,
    Timestamp, TxGraph, UpdateFailure,
};

pub type TimestampedChainGraph = ChainGraph<Option<Timestamp>>;

#[derive(Clone, Debug, Default)]
pub struct ChainGraph<E = ()> {
    chain: SparseChain<E>,
    graph: TxGraph,
}

impl<E: ChainIndexExtension> ChainGraph<E> {
    pub fn insert_tx(&mut self, tx: Transaction, index: E::IntoIndex) -> Result<bool, InsertTxErr> {
        let changed = self.chain.insert_tx(tx.txid(), index)?;
        self.graph.insert_tx(&tx);
        Ok(changed)
    }

    pub fn insert_output(
        &mut self,
        outpoint: OutPoint,
        txout: TxOut,
        index: E::IntoIndex,
    ) -> Result<bool, InsertTxErr> {
        let changed = self.chain.insert_tx(outpoint.txid, index)?;
        self.graph.insert_txout(outpoint, txout);
        Ok(changed)
    }

    pub fn insert_txid(&mut self, txid: Txid, index: E::IntoIndex) -> Result<bool, InsertTxErr> {
        self.chain.insert_tx(txid, index)
    }

    pub fn insert_checkpoint(&mut self, block_id: BlockId) -> Result<bool, InsertCheckpointErr> {
        self.chain.insert_checkpoint(block_id)
    }

    pub fn chain(&self) -> &SparseChain<E> {
        &self.chain
    }

    pub fn graph(&self) -> &TxGraph {
        &self.graph
    }

    pub fn apply_update(&mut self, update: &Self) -> Result<ChangeSet<E>, UpdateFailure<E>> {
        let changeset = self.chain.determine_changeset(update.chain())?;
        changeset
            .tx_additions()
            .map(|new_txid| update.graph.tx(new_txid).expect("tx should exist"))
            .for_each(|tx| {
                self.graph.insert_tx(tx);
            });
        self.chain.apply_changeset(&changeset);
        Ok(changeset)
    }
}
