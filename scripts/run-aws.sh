#!/usr/bin/env bash

# Set up script
set -u -o pipefail

set +e # Don't fail on error for the info-gathering section

echo "#################################################"
echo "# Environment Variables                         #"
echo "#################################################"
printenv

echo "#################################################"
echo "# CPU Information                               #"
echo "#################################################"
echo "lscpu: -----------------------------------------"
lscpu

echo "numastat: ---------------------------------------"
numastat

echo "numactl --hardware: -----------------------------"
numactl --hardware

echo "cat /proc/cpuinfo: ------------------------------"
cat /proc/cpuinfo

echo "#################################################"
echo "# PCIe Information                              #"
echo "#################################################"
lspci -tvv

echo "#################################################"
echo "# GPU Information                               #"
echo "#################################################"
echo "nvidia-smi topo -m: -----------------------------"
nvidia-smi topo -m

echo "nvidia-smi -q -d CLOCK (clock speeds): ----------"
nvidia-smi -q -d CLOCK

echo "#################################################"
echo "# Network Information                           #"
echo "#################################################"
echo "Hostname: $(hostname)\n\n"

ip a

echo "#################################################"
echo "# EFA Information                               #"
echo "#################################################"

fi_info -p efa -t FI_EP_RDM

echo "#################################################"

# Fail on error for actual experiments
set -e

# Environment
export CUDA_HOME="/usr/local/cuda"
export CUDA_PATH="/usr/local/cuda"
export EFA_PATH="/opt/amazon/efa"
export OPENMPI_PATH="/opt/amazon/openmpi"
export MPI_HOME="${OPENMPI_PATH}"
export AWS_OFI_NCCL_PATH="/mnt/sharedfs/ly-experiments/aws-ofi-nccl-lyd"
export MSCCL_PATH="/mnt/sharedfs/ly-experiments/msccl-lyd/build"
export NCCL_HOME="/mnt/sharedfs/ly-experiments/msccl-lyd/build"
export NCCL_PATH="${NCCL_HOME}"
export NCCL_TESTS_HOME="/mnt/sharedfs/ly-experiments/nccl-tests-lyd/build"
export MSCCL_XMLS="/mnt/sharedfs/ly-experiments/msccl-tools-lyd/examples/xml/xml_lyd/aws-test/32nic/32gpus"

# Config
export MPI_HOSTFILE="/home/ec2-user/hostfile"
export NUM_NODES=4
export GPUS_PER_NODE=8
export EXPERIMENTS_OUTPUT_DIR="/mnt/sharedfs/ly-experiments/experiments_output"
export LOGS_DIR="/mnt/sharedfs/ly-experiments/experiments_output/raw_logs"

# Verify that the directories exist
if [ ! -d "${CUDA_HOME}" ]; then
    echo "CUDA_HOME does not exist: ${CUDA_HOME}"
    exit 1
fi
if [ ! -d "${EFA_PATH}" ]; then
    echo "MSCCL_PATH does not exist: ${EFA_PATH}"
    exit 1
fi
if [ ! -d "${OPENMPI_PATH}" ]; then
    echo "OPENMPI_PATH does not exist: ${OPENMPI_PATH}"
    exit 1
fi
if [ ! -d "${MSCCL_PATH}" ]; then
    echo "MSCCL_PATH does not exist: ${MSCCL_PATH}"
    exit 1
fi
if [ ! -d "${NCCL_TESTS_HOME}" ]; then
    echo "NCCL_TESTS_HOME does not exist: ${NCCL_TESTS_HOME}"
    exit 1
fi
if [ ! -d "${MSCCL_XMLS}" ]; then
    echo "MSCCL_XMLS does not exist: ${MSCCL_XMLS}"
    exit 1
fi



# Print experiment information
echo "#################################################"
echo "# NCCL/MSCCL Info                               #"
echo "#################################################"
echo "NCCL Path: ${NCCL_PATH}"
echo "NCCL Commit: $(git -C ${NCCL_PATH} rev-parse --verify HEAD)"
echo "MSCCL Path: ${MSCCL_PATH}"
echo "MSCCL Commit: $(git -C ${MSCCL_PATH} rev-parse --verify HEAD)"

echo "#################################################"
echo "# MSCCL Tools/XMLs Info                         #"
echo "#################################################"
echo "MSCCL XMLs Path: ${MSCCL_XMLS}"
echo "MSCCL XMLs Commit: $(git -C ${MSCCL_XMLS} rev-parse --verify HEAD)"

echo "#################################################"

# Update Paths
export PATH="${MPI_HOME}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${AWS_OFI_NCCL_PATH}/lib:${NCCL_HOME}/lib:${MPI_HOME}/lib64:${MPI_HOME}/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Create logs directory
mkdir -p "${EXPERIMENTS_OUTPUT_DIR}"
mkdir -p "${LOGS_DIR}"

# Rust Environment
export RUST_BACKTRACE=1
export RUST_LOG=TRACE

# Whether or not to skip experiments that have already been run
# Note: You can set this to true if you don't want to re-run a bunch of experiments after a hang somewhere. You
#       will probably want to delete the "half-finished" logfiles first or those experiments will be skipped
#       as well.
export SKIP_FINISHED=FALSE

# Print commands
set -x

# Run experiments
# mpirun --hostfile ~/hostfile --map-by ppr:1:node cargo build --release  # NOT NECESSARY BECAUSE RUNS ON ONE NODE
cargo build --release
./target/release/nccl_harness 2>&1 | tee "${LOGS_DIR}/nccl_harness.$(date +%Y%m%d%H%M%S).log"

set +x
echo "Done."