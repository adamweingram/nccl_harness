#!/usr/bin/env bash

# Rust Environment
export RUST_BACKTRACE=1
export RUST_LOG=TRACE

# Setup Variables
export CUDA_HOME="/usr/local/cuda"
export CUDA_PATH="/usr/local/cuda"
export EFA_PATH="/opt/amazon/efa"
export AWS_OFI_NCCL_PATH="/opt/aws-ofi-nccl-lyd"
export OPENMPI_PATH="/opt/amazon/openmpi"
export MSCCL_PATH="/home/ec2-user/deps/msccl/build"
export NCCL_HOME="/home/ec2-user/deps/msccl/build"
export NCCL_TESTS_HOME="/home/ec2-user/deps/nccl-tests-lyd/build"
export MSCCL_XMLS="/home/Adam/Projects/NCCL-Harness/nccl-experiments/nccl_harness/msccl_tools_lyd/examples/xml/xml_lyd/aws-test/8nic/64gpus"
export MPI_HOSTFILE="/home/ec2-user/hostfile"
export NUM_NODES=8
export GPUS_PER_NODE=8
export EXPERIMENTS_OUTPUT_DIR="./logs/outputs"

# Modified here because we'll actually use the logs dir in our test
export LOGS_DIR="$(pwd)/logs"

# PERFORM A DRY RUN FOR TESTING
export DRY_RUN=TRUE

# Choose whether to skip completed experiments
export SKIP_FINISHED=TRUE

# ./target/release/test_ideas
cargo build --features no_check_paths
./target/debug/nccl_harness 2>&1 | tee "${LOGS_DIR}/dry_run.$(date +%Y%m%d%H%M%S).log"
