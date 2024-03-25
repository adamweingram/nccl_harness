# NCCL Experiments Harness
Explore NCCL's parameter space and automatically collect results.

## How to Use
0. Verify you have `cargo` installed (If you're using a cluster I set up on AWS, *then you don't need to do anything*)
    ```bash
    cargo --version
    ```
1. Edit your run script (use `run-aws.sh` for reference) and set important envvars
    ```bash
    vim ./scripts/run-aws.sh
    ```
2. Edit the experiment parameters in `main.rs` (Ctrl+f "experimental setup"). Sets of parameters are given as lists and are combined in different configurations.
    ```bash
    vim ./scripts/run-aws.sh
    # Search: / Experimental setup
    ```
3. Run using the script (the script will handle compiling for you; you don't need to run the compilation yourself):
    ```bash
    ./scripts/run-aws.sh 2>&1 | tee $OUTPUT_DIR/full-log.$(date +%Y%m%d%H%M%S).log
    ```
    Why use the `tee`? Well, the script has some extra commands that contain some useful information. This is helpful when looking back at the results.

IMPORTANT NOTE: You should not run the harness with MPI. The harness will perform the MPI call for you with the appropriate parameters.
