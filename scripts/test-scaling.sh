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
export MSCCL_PATH="/mnt/sharedfs/ly-experiments/msccl-lyd/build"
export NCCL_HOME="/mnt/sharedfs/ly-experiments/msccl-lyd/build"
export NCCL_TESTS_HOME="/mnt/sharedfs/ly-experiments/nccl-tests-lyd/build"
export MSCCL_XMLS="/home/Adam/Projects/NCCL-Harness/nccl-experiments/nccl_harness/msccl_tools_lyd/examples/xml/xml_lyd/aws-test/8nic/64gpus"
export GPUS_PER_NODE=8
export EXPERIMENTS_OUTPUT_DIR="./logs/outputs"

# Special iterable vars
export MPI_HOSTFILES=( "/home/ec2-user/hostfile-2n" "/home/ec2-user/hostfile-4n" "/home/ec2-user/hostfile-8n" )
export NUM_NODES_LIST=(2 4 8)

# Modified here because we'll actually use the logs dir in our test
export LOGS_DIR="$(pwd)/logs"

# PERFORM A DRY RUN FOR TESTING
export DRY_RUN=TRUE

# Choose whether to skip completed experiments
export SKIP_FINISHED=FALSE

# Build the harness
cargo build --features no_check_paths

# Run experiments
for hostfile in "${MPI_HOSTFILES[@]}"; do
    for num_nodes in "${NUM_NODES_LIST[@]}"; do
        echo "Running with hostfile: ${hostfile}, num_nodes: ${num_nodes}"

        # Set envvars required by the harness
        export MPI_HOSTFILE="${hostfile}"
        export NUM_NODES="${num_nodes}"
        echo "Will use MPI_HOSTFILE=${MPI_HOSTFILE}, NUM_NODES=${NUM_NODES}"

        # Run the harness
        ./target/debug/nccl_harness 2>&1 | tee "${LOGS_DIR}/dry_run_scaling-${num_nodes}node.$(date +%Y%m%d%H%M%S).log"
    done
done

echo "Done with all experiments"