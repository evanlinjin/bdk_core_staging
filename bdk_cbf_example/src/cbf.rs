use std::{collections::HashSet, net, thread};

use nakamoto::{
    client::{
        chan, network::Services, traits::Handle, Client, Config, Domain, Event,
        Handle as ClientHandle, Network,
    },
    net::poll,
};

use bdk_cli::{anyhow, Keychain};

use bdk_chain::{
    bitcoin::{blockdata::script::Script, Transaction},
    chain_graph::ChainGraph,
    keychain::{KeychainChangeSet, KeychainTracker},
    BlockId, TxHeight,
};

type Reactor = poll::Reactor<net::TcpStream>;

// Things to handle:
// Block disconnected before filter processed, what to do?
// I'm not saving the txouts and spk_txouts in the SpkTxOutIndex in KeychainTxOutIndex

pub struct CbfClient {
    pub handle: ClientHandle<poll::reactor::Waker>,
    pub client_recv: chan::Receiver<Event>,
}

impl CbfClient {
    pub fn new(network: Network) -> anyhow::Result<Self> {
        let mut cfg = Config::new(network);
        cfg.domains = vec![Domain::IPV4];
        let client = Client::<Reactor>::new()?;
        let handle = client.handle();
        let client_recv = handle.events();

        // let genesis = handle.get_block_by_height(0).expect("whattt").expect("should be there?");
        // let hash = genesis.block_hash();
        // println!("Genesis Block Hash: {}", hash);

        // Run the client on a different thread, to not block the main thread.
        thread::spawn(|| client.run(cfg).unwrap());

        println!("Looking for peers...");
        // Wait for the client to be connected to a peer.
        handle.wait_for_peers(1, Services::default())?;
        println!("Connected to at least one peer");

        Ok(CbfClient {
            handle,
            client_recv,
        })
    }

    pub fn sync(
        &mut self,
        keychain_tracker: &mut KeychainTracker<Keychain, TxHeight>,
        scripts: impl Iterator<Item = Script>,
        stop_gap: u32,
    ) -> anyhow::Result<KeychainChangeSet<Keychain, TxHeight>> {
        let mut processed_height = keychain_tracker
            .chain_graph()
            .chain()
            .latest_checkpoint()
            .map(|c| c.height)
            .unwrap_or(63000) as u64; // TODO
        println!("Rescanning chain from {:?}", processed_height);
        self.handle.rescan(processed_height.., scripts)?;
        let mut blocks_matched = HashSet::new();
        let mut peer_height = 0;
        let mut update = ChainGraph::<TxHeight>::default();
        let mut txs = vec![];

        loop {
            chan::select! {
                recv(self.client_recv) -> event => {
                    let event = event?;
                    match event {
                        Event::PeerNegotiated { height, .. } => {
                            println!("Peer negotiated with height {:?}", height);
                            if peer_height < height {
                                peer_height = height;
                            }
                            if processed_height == peer_height {
                                // TODO: improve this!
                                // It might be that both me and this peer
                                // are lagging behind
                                break;
                            }
                        }
                        Event::PeerHeightUpdated { height, .. } => {
                            if peer_height < height {
                                peer_height = height;
                            }
                        }
                        Event::BlockConnected { height, .. } => {
                            if height % 1000 == 0 {
                                println!("Connected block with height {:?}", height);
                            }
                        }
                        Event::BlockDisconnected { height, hash, .. } => {
                            println!("Disconnected block with height {:?}", height);
                            // TODO: what happens if a block gets disconnected before I process its
                            // filter?
                            let _ = update.invalidate_checkpoints(height as u32);
                            blocks_matched.remove(&hash);
                        }
                        Event::BlockMatched { height, hash, transactions, .. } => {
                            println!("Block matched {:?}", height);
                            for tx in transactions {
                                txs.push((tx, TxHeight::Confirmed(height as u32)));

                            }
                            blocks_matched.remove(&hash);
                            if processed_height >= peer_height && blocks_matched.is_empty() {
                                break;
                            }
                        }
                        Event::TxStatusChanged { .. } => {
                            println!("Tx status changed {:?}", &event);
                        }
                        Event::FilterProcessed { matched, height, block: hash, .. } => {
                            let _ = update.insert_checkpoint(BlockId { height: height as u32, hash })?;
                            if height % 1000 == 0 {
                                println!("Filter processed {}", height);
                            }
                            processed_height = height;
                            if matched {
                                println!("Filter matched {:?}", &event);
                                blocks_matched.insert(hash);
                            }

                            if processed_height == peer_height && blocks_matched.is_empty() {
                                break;
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        let old_indexes = keychain_tracker.txout_index.derivation_indices();

        for (tx, _) in &txs {
            keychain_tracker.txout_index.pad_all_with_unused(stop_gap);
            keychain_tracker.txout_index.scan(tx);
        }

        // inserting only relevant txs in the txgraph
        for (tx, height) in txs {
            if keychain_tracker.txout_index.is_relevant(&tx) {
                println!("* adding tx to update: {} @ {}", tx.txid(), height);
                let _ = update.insert_tx(tx.clone(), height)?;
            }
        }

        // @evanlinjin: @danielabrozzoni, the culprit is here!
        let new_indexes = keychain_tracker.txout_index.last_active_indicies();

        keychain_tracker
            .txout_index
            .prune_unused(old_indexes.clone());

        let changeset = KeychainChangeSet {
            derivation_indices: keychain_tracker
                .txout_index
                .keychains()
                .keys()
                .filter_map(|keychain| {
                    let old_index = old_indexes.get(keychain);
                    let new_index = new_indexes.get(keychain);

                    match new_index {
                        Some(new_ind) if new_index > old_index => {
                            Some((keychain.clone(), *new_ind))
                        }
                        _ => None,
                    }
                })
                .collect(),
            chain_graph: keychain_tracker
                .chain_graph()
                .determine_changeset(&update)?,
        };

        println!("changeset: {:#?}", changeset);
        // dbg!(&changeset.chain_graph.graph.txout);
        Ok(changeset)
    }
}

impl bdk_cli::Broadcast for CbfClient {
    type Error = nakamoto::client::handle::Error;
    fn broadcast(&self, tx: &Transaction) -> Result<(), Self::Error> {
        self.handle.submit_transaction(tx.clone())?;
        Ok(())
    }
}
