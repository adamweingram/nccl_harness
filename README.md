# NCCL Experiments Harness
Explore NCCL's parameter space and automatically collect results.

## How to Use
1. Edit your run script (use `run-aws.sh` for reference) and set important envvars
2. Edit the experiment parameters in `main.rs` (Ctrl+f "experimental setup")
3. Build and run using the script (the script will handle compiling for you)

IMPORTANT NOTE: You should not run the harness with MPI. The harness will perform the MPI call for you with the appropriate parameters.
