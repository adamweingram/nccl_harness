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

# Load Deps
module load cuda
spack load gcc@10.4.0%gcc@8.5.0 arch=linux-almalinux8-icelake
# spack load openmpi
module load mpich-4.0.2-gcc-8.5.0-atqvq3l

# Print info
nvidia-smi topo -m

# Set variables
export CUDA_HOME="/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda"
export CUDA_PATH="/opt/nvidia/hpc_sdk/Linux_x86_64/21.9/cuda"
export NCCL_HOME="/home/Adam/Projects/NCCL-Harness/nccl-experiments/nccl/build"
# export MPI_HOME="/home/Adam/Software/spack/opt/spack/linux-almalinux8-icelake/gcc-8.5.0/openmpi-4.1.4-3qxzgu5b6g3yyftfcz2dq7pxjostt4b6"
export MPI_HOME="$(dirname $(dirname $(which mpirun)))"
export NCCL_TESTS_HOME="/home/Adam/Projects/NCCL-Harness/nccl-experiments/nccl-tests/build"
export EXPERIMENTS_OUTPUT_DIR="/home/Adam/Projects/NCCL-Harness/nccl-experiments/nccl_harness/experiments_output"

# Update Paths
export PATH="${MPI_HOME}/bin:${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${MPI_HOME}/lib:${MPI_HOME}/lib64:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Run experiments
cargo build --release
./target/release/nccl_harness | tee "logs/nccl_harness.log"

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