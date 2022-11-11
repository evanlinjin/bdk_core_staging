use std::time::SystemTime;

use bdk_core::{
    bitcoin::{
        hashes::Hash, secp256k1::Secp256k1, Address, BlockHash, Network, OutPoint, PackedLockTime,
        Script, Sequence, Transaction, TxIn, TxOut, Witness,
    },
    miniscript::{Descriptor, DescriptorPublicKey},
    BlockId, ConfirmationTime, Timestamp, TimestampedChainGraph, TxHeight,
};

const DESC: &str = "tr(xprv9uBuvtdjghkz8D1qzsSXS9Vs64mqrUnXqzNccj2xcvnCHPpXKYE1U2Gbh9CDHk8UPyF2VuXpVkDA7fk5ZP4Hd9KnhUmTscKmhee9Dp5sBMK)";

fn main() {
    let mut wallet = TimestampedChainGraph::default();

    let secp = Secp256k1::new();
    let (desc, _) = Descriptor::<DescriptorPublicKey>::parse_descriptor(&secp, DESC).unwrap();

    let mine = desc
        .at_derivation_index(0)
        .address(Network::Testnet)
        .unwrap();
    let other = desc
        .at_derivation_index(1)
        .address(Network::Testnet)
        .unwrap();

    let start = SystemTime::now();

    for height in 1_u32..100_000 {
        let txs = prepare_txs(height, 10, &mine, &other);
        let update = prepare_update(height - 1, height, txs);
        wallet.apply_update(&update).expect("should succeed");
    }

    println!(
        "[sync done] elapsed: {}s, 100_000 blocks, 10 txs per block",
        start.elapsed().unwrap().as_secs()
    );

    let sum = wallet
        .graph()
        .iter_all_txouts()
        .filter(|(_, txo)| txo.script_pubkey == mine.script_pubkey())
        .map(|(_, txo)| txo.value)
        .sum::<u64>();

    println!(
        "[got balance] elapsed: {}s, balance: {}sats",
        start.elapsed().unwrap().as_secs(),
        sum
    );
}

fn prepare_update(cp_old: u32, cp_new: u32, txs: Vec<Transaction>) -> TimestampedChainGraph {
    let mut update = TimestampedChainGraph::default();

    for height in [cp_old, cp_new] {
        let hash = BlockHash::hash(&height.to_be_bytes());
        let block_id = BlockId { height, hash };
        update.insert_checkpoint(block_id).expect("should succeed");
    }

    for tx in txs {
        update
            .insert_tx(
                tx,
                ConfirmationTime {
                    height: TxHeight::Confirmed(cp_new),
                    time: Timestamp(cp_new as _),
                },
            )
            .expect("should succeed");
    }

    update
}

fn prepare_txs(height: u32, tx_count: usize, mine: &Address, other: &Address) -> Vec<Transaction> {
    (0..tx_count)
        .map(|i| Transaction {
            version: 0,
            lock_time: PackedLockTime::ZERO,
            input: vec![
                TxIn {
                    previous_output: OutPoint::new(Hash::hash(&(height - 1).to_le_bytes()), 0),
                    script_sig: Script::new(),
                    sequence: Sequence::from_consensus(i as _),
                    witness: Witness::new(),
                },
                TxIn {
                    previous_output: OutPoint::new(Hash::hash(&(height - 1).to_le_bytes()), 1),
                    script_sig: Script::new(),
                    sequence: Sequence::from_consensus(i as _),
                    witness: Witness::new(),
                },
            ],
            output: vec![
                TxOut {
                    value: 100_000,
                    script_pubkey: mine.script_pubkey(),
                },
                TxOut {
                    value: 200_000,
                    script_pubkey: other.script_pubkey(),
                },
            ],
        })
        .collect()
}
