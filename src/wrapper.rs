use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use log::{debug, info, warn, error};

use crate::{Row, Permutation, MscclExperimentParams};

/// Run NCCL tests with MPI using a set of parameters
pub fn run_msccl_tests(
    executable: &Path,
    exp_params: &MscclExperimentParams,
    ignore_error_status_codes: bool,
    dry_run: bool,
    output_path: Option<PathBuf>,
    stderr_path: Option<PathBuf>,
) -> Result<Vec<Row>, Box<dyn std::error::Error>> {
    // Build the LD_LIBRARY_PATH from the given environment variables
    let mut ld_library_path = format!(
        "{}/lib64:{}/lib:{}/lib64:{}/lib:{}/lib64:{}/lib",
        exp_params.cuda_path,
        exp_params.cuda_path,
        exp_params.openmpi_path,
        exp_params.openmpi_path,
        exp_params.msccl_path,
        exp_params.msccl_path
    );
    if let Some(efa_path) = exp_params.efa_path.clone() {
        ld_library_path.push_str(format!(":{}/lib", efa_path).as_str());
    }
    if let Some(aws_ofi_nccl_path) = exp_params.aws_ofi_nccl_path.clone() {
        ld_library_path.push_str(format!(":{}/lib", aws_ofi_nccl_path).as_str());
    }
    debug!("Will use `LD_LIBRARY_PATH`: {}", ld_library_path);

    // MSCCL XML file handling (just use dummy envvar if not given an XML file)
    let msccl_xml_envvar = {
        debug!(
            "Using MSCCL XML file at: {}",
            exp_params.ms_xml_file.to_str().unwrap()
        );
        format!(
            "MSCCL_XML_FILES={}",
            exp_params.ms_xml_file.to_str().unwrap()
        )
    };

    // Run NCCL tests with MPI
    // TODO: Verify that OpenMPI passes through required environment variables
    debug!("Running NCCL tests with 'MPI'...");
    if dry_run {
        info!("🌵 ONLY PRINTING OUT THE COMMAND BECAUSE THIS IS A DRY RUN! 🌵")
    }
    let mut res = Command::new(if !dry_run { "mpirun" } else { "echo" })
        .args(["--hostfile", exp_params.mpi_hostfile_path.to_str().unwrap()])
        .args([
            "--map-by",
            format!("ppr:{}:node", exp_params.mpi_proc_per_node).as_str(),
        ])
        .args([
            "-x",
            format!("LD_LIBRARY_PATH={}", ld_library_path).as_str(),
        ])
        .args(["-x", msccl_xml_envvar.as_str()])
        .args(["-x", "GENMSCCLXML=1"])
        .args([
            "-x",
            format!("NCCL_DEBUG={}", exp_params.nccl_debug_level).as_str(),
        ])
        .args(["-x", format!("NCCL_ALGO={}", exp_params.nccl_algo).as_str()])
        .args(["-x", "FI_EFA_USE_DEVICE_RDMA=1"])
        .args(["-x", "FI_EFA_FORK_SAFE=1"])
        .args([
            "--mca",
            "btl",
            "tcp,self",
            "--mca",
            "btl_tcp_if_exclude",
            "lo,docker0",
            "--bind-to",
            "none",
        ])
        .arg(executable.to_str().unwrap())
        .args([
            "--nthreads",
            format!("{}", exp_params.nc_num_threads).as_str(),
        ])
        .args(["--ngpus", exp_params.nc_num_gpus.to_string().as_str()])
        .args(["--minbytes", exp_params.nc_min_bytes.as_str()])
        .args(["--maxbytes", exp_params.nc_max_bytes.as_str()])
        .args(["--stepfactor", exp_params.nc_step_factor.as_str()])
        .args(["--op", exp_params.nc_op.as_str()])
        .args(["--datatype", exp_params.nc_dtype.as_str()])
        .args(["--iters", exp_params.nc_num_iters.to_string().as_str()])
        .args([
            "--warmup_iters",
            exp_params.nc_num_warmup_iters.to_string().as_str(),
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("[ERROR] FAILED TO RUN WITH MPI!!!!");

    // Create vector to store rows
    let mut rows = Vec::new();

    // Open output file for writing
    let mut output_file = match output_path {
        Some(path) => {
            debug!("Opening output file at: {}", path.to_str().unwrap());
            let res = match std::fs::File::create(path.clone()) {
                Ok(f) => Some(f),
                Err(e) => {
                    error!("Error creating output file {:?}. WILL NOT LOG OUTPUT AS SEPARATE FILE!: {}", path, e);
                    None
                }
            };

            res
        }
        None => None,
    };

    let mut stderr_file = match stderr_path {
        Some(path) => {
            debug!("Opening stderr file at: {}", path.to_str().unwrap());
            let res = match std::fs::File::create(path.clone()) {
                Ok(f) => Some(f),
                Err(e) => {
                    error!("Error creating stderr file {:?}. WILL NOT LOG STDERR AS SEPARATE FILE!: {}", path, e);
                    None
                }
            };

            res
        }
        None => None,
    };

    // Print and handle stdout line by line
    let stdout_reader = std::io::BufReader::new(res.stdout.take().unwrap());
    for line in stdout_reader.lines() {
        match line {
            Ok(line) => {
                debug!("[l]: {}", line);

                // Write to output file
                if let Some(file) = &mut output_file {
                    match file.write_all(line.as_bytes()) {
                        Ok(_) => {}
                        Err(e) => {
                            error!("Error writing line to output file: {}", e);
                        }
                    
                    };
                    
                    match file.write_all(b"\n") {
                        Ok(_) => {}
                        Err(e) => {
                            error!("Error writing newline to output file: {}", e);
                        }
                    };
                }

                // Parse line
                // TODO: Add function when stable
            }
            Err(e) => {
                error!("Error getting line from stdout BufReader: {}", e);
            }
        }
    }

    // Print stderr
    // FIXME: Won't actually print if there's a hang-related error! The stdout reader never finishes reading!
    let stderr_reader = std::io::BufReader::new(res.stderr.take().unwrap());
    for line in stderr_reader.lines() {
        match line {
            Ok(line) => {
                // Print the line
                debug!("[E]: {}", line);

                // Write to stderr file
                if let Some(file) = &mut stderr_file {
                    match file.write_all(line.as_bytes()) {
                        Ok(_) => {}
                        Err(e) => {
                            error!("Error writing line to stderr file: {}", e);
                        }
                    
                    };
                    
                    match file.write_all(b"\n") {
                        Ok(_) => {}
                        Err(e) => {
                            error!("Error writing newline to stderr file: {}", e);
                        }
                    };
                }
            }
            Err(e) => {
                error!("Error getting line from stderr BufReader: {}", e);
            }
        }
    }

    // Handle exit status
    let status = res.wait()?;
    match status.success() {
        true => info!("[SUCCESS] NCCL tests with MPI ran successfully."),
        false => {
            if !ignore_error_status_codes {
                error!(
                    "Running NCCL tests with MPI failed with exit code: {}",
                    status.code().unwrap()
                );
                return Err("NCCL tests with MPI failed.".into());
            } else {
                error!(
                    "Running NCCL tests with MPI failed with exit code: {}, but ignoring and continuing.",
                    status.code().unwrap()
                );
            }
        }
    }

    Ok(rows)
}