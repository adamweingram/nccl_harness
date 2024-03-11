#!/usr/bin/env bash

#SBATCH --job-name=nccl-tests
#SBATCH --partition=A100
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --nodelist=node01,node02
#SBATCH --output=logs/nccl-test.%j.log

# Set up script
set -e

# Print info
nvidia-smi topo -m

# Set variables
export CUDA_HOME="/home/ec2-user/deps/cuda"
export CUDA_PATH="/home/ec2-user/deps/cuda"
# export NCCL_HOME="/home/ec2-user/deps/nccl/build"
export NCCL_HOME="/home/ec2-user/deps/msccl/build"
export MPI_HOME="$(dirname $(dirname $(which mpirun)))"
# export NCCL_TESTS_HOME="/home/Adam/Projects/NCCL-Harness/nccl-experiments/nccl-tests/build"
export NCCL_TESTS_HOME="/home/ec2-user/deps/msccl-tests/build"
export EXPERIMENTS_OUTPUT_DIR="/home/ec2-user/experiments_output"
export LOGS_DIR="/home/ec2-user/logs"
export MPI_HOSTFILE="/home/ec2-user/hostfile"

# Update Paths
export PATH="${MPI_HOME}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${MPI_HOME}/lib:${MPI_HOME}/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Run experiments
cargo build --release
./target/release/nccl_harness | tee "${LOGS_DIR}/nccl_harness.log"

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