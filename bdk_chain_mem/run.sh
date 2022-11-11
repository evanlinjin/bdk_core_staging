RUSTFLAGS='-C force-frame-pointers=y' cargo build --release -p bdk_chain_mem

# ./target/release/bdk_chain_mem
perf record -g ./target/release/bdk_chain_mem
