#!/usr/bin/env bash

# Set up script
set -e -u -o pipefail

echo "#################################################"
echo "# Environment Variables                         #"
echo "#################################################"
printenv
echo "#################################################"

# Print info
nvidia-smi topo -m

# Set variables
export CUDA_HOME="/usr/local/cuda"
export CUDA_PATH="/usr/local/cuda"
export EFA_PATH="/opt/amazon/efa"
export AWS_OFI_NCCL_PATH="/opt/aws-ofi-nccl-lyd"
export OPENMPI_PATH="/opt/amazon/openmpi"
export MSCCL_PATH="/home/ec2-user/ly-custom/msccl-lyd/build"
export NCCL_HOME="/home/ec2-user/ly-custom/msccl-lyd/build"
export NCCL_TESTS_HOME="/home/ec2-user/ly-custom/nccl-tests-lyd/build"
export MSCCL_XMLS="/home/ec2-user/ly-custom/msccl-tools-lyd/examples/xml/xml_lyd/aws-test/8nic/16gpus"
export MPI_HOSTFILE="/home/ec2-user/hostfile"
export NUM_NODES=8
export GPUS_PER_NODE=8
export EXPERIMENTS_OUTPUT_DIR="/home/ec2-user/experiments_output"
export LOGS_DIR="/home/ec2-user/experiments_output/raw_logs"

# Update Paths
export PATH="${MPI_HOME}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${NCCL_PLUGIN_LIBS}:${NCCL_HOME}/lib:${MPI_HOME}/lib64:${MPI_HOME}/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Create logs directory
mkdir -p "${EXPERIMENTS_OUTPUT_DIR}"
mkdir -p "${LOGS_DIR}"

# Run experiments
# mpirun --hostfile ~/hostfile --map-by ppr:1:node cargo build --release  # NOT NECESSARY BECAUSE RUNS ON ONE NODE
cargo build --release
./target/release/nccl_harness 2>&1 | tee "${LOGS_DIR}/nccl_harness.$(date +%Y%m%d%H%M%S).log"

# mpirun --hostfile ~/hostfile --map-by ppr:8:node \
#     -x LD_LIBRARY_PATH \
#     -x NCCL_DEBUG="INFO" \
#     -x FI_EFA_FORK_SAFE=1 \
#     -x MSCCL_XML_FILES="/home/ec2-user/deps/msccl-tools/examples/xml/xml_lyd/aws-test/1nic/allreduce_binary_tree_1ch_64chunk.xml" \
#     --mca btl tcp,self --mca btl_tcp_if_exclude lo,docker0 --bind-to none \
#     /home/ec2-user/deps/nccl-tests-base/build/all_reduce_perf \
#         --minbytes 512 \
#         --maxbytes 128M \
#         -f 2 -g 1

# mpirun \
#     -host node01,node02 \
#     --map-by ppr:1:node \
#     --mca btl self,vader,tcp \
#     -x LD_LIBRARY_PATH \
#     -x PATH \
#     -x CUDA_HOME \
#     -x CUDA_PATH \
#     -x NCCL_HOME \
#     -x MPI_HOME \
#     hostname

# mpirun \
#     -host node01,node02 \
#     --map-by ppr:1:node \
#     hostname

echo "Done."