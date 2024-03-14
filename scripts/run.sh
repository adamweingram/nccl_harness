#!/usr/bin/env bash

#SBATCH --job-name=nccl-tests
#SBATCH --partition=A100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node01,node02
#SBATCH --output=logs/nccl-test.%j.log

# Set up script
set -e -u -o pipefail

# Print info
nvidia-smi topo -m

# Set variables
export CUDA_HOME="/usr/local/cuda"
export CUDA_PATH="/usr/local/cuda"
export NCCL_HOME="/home/ec2-user/deps/msccl/build"
export MPI_HOME="$(dirname $(dirname $(which mpirun)))"
export NCCL_PLUGIN_LIBS="/opt/aws-ofi-nccl/lib"
export NCCL_TESTS_HOME="/home/ec2-user/deps/msccl-tests/build"
export EXPERIMENTS_OUTPUT_DIR="/home/ec2-user/experiments_output"
export LOGS_DIR="/home/ec2-user/nccl_harness/logs"
export MPI_HOSTFILE="/home/ec2-user/hostfile"
export MSCCL_XMLS="/home/ec2-user/deps/msccl-tools/examples/xml/xml_lyd/aws-test/1nic"
export GPUS_PER_NODE="4"

# Update Paths
export PATH="${MPI_HOME}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${NCCL_PLUGIN_LIBS}:${NCCL_HOME}/lib:${MPI_HOME}/lib64:${MPI_HOME}/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Run experiments
# mpirun --hostfile ~/hostfile --map-by ppr:1:node cargo build --release  # NOT NECESSARY BECAUSE RUNS ON ONE NODE
cargo build --release
./target/release/nccl_harness | tee "${LOGS_DIR}/nccl_harness.log"

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