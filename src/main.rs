use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::Command;
use regex::Regex;
use polars::prelude::*;
use log::{debug, info, warn, error};
#[macro_use] extern crate prettytable;

mod util;
use util::{Row, Permutation, MscclExperimentParams, ManifestEntry, ResultDescription, params_to_xml, verify_env, pretty_print_configs, pretty_print_result_manifest, collective_to_test_exe};

mod parse;
use parse::{rows_to_df, parse_line};

mod wrapper;
use wrapper::run_msccl_tests;

use crate::util::exp_params_to_output_filename;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();

    // CUDA Path
    let cuda_path = match std::env::var("CUDA_HOME") {
        Ok(v) => {
            debug!("CUDA_HOME set to: {}", v);
            v
        },
        Err(_) => {
            panic!("[ERROR] CUDA_HOME not set!");
        }
    };

    // EFA Path
    let efa_path = match std::env::var("EFA_PATH") {
        Ok(v) => {
            debug!("EFA_PATH set to: {}", v);
            Some(v)
        },
        Err(_) => {
            warn!("EFA_PATH was not set! You will not be able to run tests that use the EFA!");
            None
        }
    };

    // AWS OFI NCCL Path
    let aws_ofi_nccl_path = match std::env::var("AWS_OFI_NCCL_PATH") {
        Ok(v) => {
            debug!("AWS_OFI_NCCL_PATH set to: {}", v);
            Some(v)
        },
        Err(_) => {
            warn!("AWS_OFI_NCCL_PATH was not set! You will not be able to run tests that use the EFA!");
            None
        }
    };

    // OpenMPI Path
    let openmpi_path = match std::env::var("OPENMPI_PATH") {
        Ok(v) => {
            debug!("OPENMPI_PATH set to: {}", v);
            v
        },
        Err(_) => {
            panic!("[ERROR] Envvar OPENMPI_PATH not set!");
        }
    };

    // MSCCL Path
    let msccl_path = match std::env::var("MSCCL_PATH") {
        Ok(v) => {
            debug!("MSCCL_PATH set to: {}", v);
            v
        },
        Err(_) => {
            panic!("[ERROR] Envvar MSCCL_PATH not set!");
        }
    };

    // NCCL tests executable binary location
    let nccl_test_bins = match std::env::var("NCCL_TESTS_HOME") {
        Ok(v) => {
            debug!("NCCL_TESTS_HOME set to: {}", v);
            PathBuf::from(v)
        },
        Err(_) => {
            panic!("[ERROR] Envvar NCCL_TESTS_HOME not set!");
        }
    };

    // MSCCL XML files location
    let msccl_xmls_directory = match std::env::var("MSCCL_XMLS") {
        Ok(v) => {
            debug!("MSCCL_XMLS set to: {}", v);
            PathBuf::from(v)
        },
        Err(_) => {
            panic!("[ERROR] Envvar MSCCL_XMLS not set!");
        }
    };

    // MPI hostfile
    let mpi_hostfile_path = match std::env::var("MPI_HOSTFILE") {
        Ok(v) => {
            debug!("MPI_HOSTFILE set to: {}", v);
            PathBuf::from(v)
        },
        Err(_) => {
            panic!("[ERROR] Envvar MPI_HOSTFILE not set!");
        }
    
    };

    #[cfg(not(feature = "no_check_paths"))]
    if !mpi_hostfile_path.exists() {
        panic!(
            "[ERROR] Envvar MPI_HOSTFILE not found at: {}",
            mpi_hostfile_path.to_str().unwrap()
        );
    }

    // Number of Nodes
    let num_nodes = match std::env::var("NUM_NODES") {
        Ok(v) => {
            debug!("NUM_NODES set to: {}", v);
            v.parse::<u64>().unwrap()
        },
        Err(_) => {
            panic!("[ERROR] Envvar NUM_NODES not set!");
        }
    };

    // GPUs per Node
    let gpus_per_node = match std::env::var("GPUS_PER_NODE") {
        Ok(v) => {
            debug!("GPUS_PER_NODE set to: {}", v);
            v.parse::<u64>().unwrap()
        },
        Err(_) => {
            panic!("[ERROR] Envvar GPUS_PER_NODE not set!");
        }
    };

    // Experiments Output Directory
    let experiments_output_dir = match std::env::var("EXPERIMENTS_OUTPUT_DIR") {
        Ok(v) => {
            debug!("EXPERIMENTS_OUTPUT_DIR set to: {}", v);
            let path = PathBuf::from(v);

            // Verify that the directory exists. Otherwise, create it.
            if !path.exists() {
                std::fs::create_dir(path.as_path())?;
                debug!("Created experiment log output directory at: {:?}", path);
            } else {
                debug!("Experiment log output directory already exists at: {:?}", path);
            }

            path
        }
        Err(_) => {
            panic!("[ERROR] Envvar EXPERIMENTS_OUTPUT_DIR not set!");
        }
    };

    // Check if should skip previously completed experiments (ala makefile)
    let skip_finished = match std::env::var("SKIP_FINISHED") {
        Ok(v) => {
            if v.to_lowercase() == "true" || v.to_lowercase() == "1" {
                info!("⏭️ Found 'SKIP_FINISHED=true', will skip experiments that already have an output file! ⏭️");
                true
            } else {
                info!("Found 'SKIP_FINISHED=false', will NOT skip experiments that already have an output file.");
                false
            }
        }
        Err(_) => {
            info!("Did not find a setting for 'SKIP_FINISHED', will NOT skip experiments that already have an output file.");
            false
        }
    };

    // Check if doing a dry run
    let dry_run = match std::env::var("DRY_RUN") {
        Ok(v) => {
            if v.to_lowercase() == "true" || v.to_lowercase() == "1" {
                info!("🌵🌵🌵 PERFORMING DRY RUN!!!! 🌵🌵🌵");
                true
            } else {
                false
            }
        }
        Err(_) => false
    };

    // Experimental setup
    // Independent Variables:
    // • Collective Algorithm (e.g., all_reduce_perf, all_gather_perf, alltoall_perf, broadcast_perf, gather_perf, hypercube_perf, reduce_perf, reduce_scatter_perf, scatter_perf, sendrecv_perf)
    // • Operation (e.g., sum, prod, min, max, avg)
    // • Data Type (e.g., double, float, int32, int8)
    // • Communication Algorithm (e.g., binary_tree, binomial_tree, recursive_doubling, recursive_doubling_halving, ring, trinomial_tree)
    // • Number of MSCCL Channels (e.g., 1, 2, 4, 8)
    // • MSCCL Chunk Size (e.g., 8, 16, 32, 64, 128, 256)
    // • Number of GPUs (count, e.g., 1, 2, 4, 8, 16, 32, 64, 128, 256)
    // • Buffer Size (scaling factor, e.g., 1, 2, 4)
    // • Message Size (bytes, e.g., 64k, 256M)

    // Hardware details
    let num_gpus = num_nodes * gpus_per_node;

    // Selected
    let num_repetitions = 2;
    let collectives = [
        "all-reduce",
        // "all-gather",
        // "all-to-all",
        // "broadcast",
        // "gather",
        // "hypercube",  // BROKEN FOR HYPERCUBE BECAUSE THE OUTPUT TABLE IS BLANK FOR REDOP (breaks parsing)
        // "reduce",
        // "reduce-scatter",
        // "scatter",
        // "sendrecv"
    ];
    let reduction_ops = [
        "sum",
        // "prod",
        // "min",
        // "max",
        // "avg"
    ];
    let data_types = [
        // "double",
        "float",
        // "int32",
        // "int8",
    ];
    let comm_algorithms = [
        "binary-tree",
        // "binomial-tree",
        // "recursive-doubling",
        // "recursive-halving-doubling",
        "ring",
        // "trinomial-tree"
    ];

    // Note: These will be determined by the special case generator in the loop (at Ly's request)
    // let msccl_potential_channels = [  // NOTE: HANDLED IN THE PERMUTATION GENERATOR BECAUSE THERE ARE SPECIAL CASES!
    //     4,
    //     8,
    //     16,
    // ];
    // let msccl_potential_chunks = [  // NOTE: HANDLED IN THE PERMUTATION GENERATOR BECAUSE THERE ARE SPECIAL CASES!
    //     1,
    //     4,
    //     16,
    //     // 64,
    //     // 256
    // ];

    // IMPORTANT: Buffer size must be modified by changing NCCL code at the moment! Therefore, we won't use
    //            the harness to select buffer sizes. We will run the harness manually three times.
    let buffer_sizes = [
        // 1u64, 
        2u64, 
        // 4u64,
    ];
    let message_size_range = ("64K", "16G"); // We use a range for all experiments
    let gpus_as_nodes = [
        // true, 
        false
    ];

    // Blacklist certain XML files that hang or otherwise misbehave
    let blacklist: [&str; 0] = [];  // Use this if you want the blacklist to contain nothing
    // let blacklist = [
    //     PathBuf::from("allreduce_ring_node4_gpu32_mcl4_mck2_gan0.xml"),
    //     PathBuf::from("allreduce_ring_node4_gpu32_mcl8_mck2_gan0.xml"),
    //     PathBuf::from("allreduce_ring_node4_gpu32_mcl16_mck2_gan0.xml"),
    // ];

    let nccl_debug_level = "INFO"; // Use `TRACE` for replayable trace information on every call

    // Store list of all experiment permutations
    let mut permutations = Vec::new();
    let mut experiment_descriptors = Vec::new();

    // Create permutations
    for collective in collectives {
        // Build executable path
        let collective_exe = collective_to_test_exe(collective)?;
        let nccl_test_executable = nccl_test_bins.join(collective_exe.clone());

        #[cfg(not(feature = "no_check_paths"))]
        assert!(nccl_test_executable.exists());

        // Run experiments across all variations
        for buffer_size in buffer_sizes {
            for data_type in data_types {
                for reduction_op in reduction_ops {
                    for comm_algorithm in comm_algorithms {
                        // Handle special cases for different communication algorithms
                        let (msccl_potential_chunks, msccl_potential_channels) =
                            match comm_algorithm {
                                "binary-tree" => (vec![1u64, 2, 4, 8, 16], vec![4u64, 8, 16]),
                                // "binomial-tree" => (vec![8, 16, 32, 64, 128], vec![1, 2]),
                                // "recursive-doubling-halving" => (vec![8, 16, 32], vec![1, 2]),
                                "ring" => (vec![1u64, 2], vec![4u64, 8, 16]),
                                // "double-binary-tree" => (vec![8, 16, 32, 64, 128, 256], vec![1, 2]),
                                // "double-binomial-tree" => (vec![8, 16, 32, 64, 128], vec![1, 2]),
                                // "trinomial-tree" => (vec![8, 16, 32, 64, 128], vec![1, 2]),
                                // "recursive-doubling" => (vec![8, 16, 32], vec![1, 2]),
                                _ => panic!("[ERROR] Unknown comm_algorithm: {}", comm_algorithm),
                            };

                        // Create permutations
                        for msccl_chunks in msccl_potential_chunks.iter() {
                            for msccl_channels in msccl_potential_channels.iter() {
                                for gpu_as_node in gpus_as_nodes {
                                    // Figure out the name of potential the XML file name for this experiment
                                    let xml_file_name = params_to_xml(
                                        collective,
                                        comm_algorithm,
                                        num_nodes,
                                        num_gpus.clone(),
                                        msccl_channels.clone(),
                                        msccl_chunks.clone(),
                                        gpu_as_node,
                                    )?;

                                    let xml_file = msccl_xmls_directory.join(xml_file_name);

                                    // Verify that the XML file exists
                                    // Note: We want to fail early if the XML file is not found rather than failing mid-way through
                                    //       running the experiments.
                                    
                                    if !xml_file.exists() {
                                        #[cfg(feature = "no_check_paths")]
                                        warn!("During permutation generation, XML file not found at: {}. Continuing because 'no_check_paths' cfg is set", xml_file.to_str().unwrap());

                                        #[cfg(not(feature = "no_check_paths"))]
                                        panic!("During permutation generation, XML file not found at: {}. Quitting.", xml_file.to_str().unwrap());
                                    } else {
                                        debug!("Found XML file at: {}", xml_file.to_str().unwrap());
                                    }

                                    // Create a full set of experiment parameters for this permutation
                                    let experiment = MscclExperimentParams {
                                        // Environment params
                                        cuda_path: cuda_path.clone(),
                                        efa_path: efa_path.clone(),
                                        aws_ofi_nccl_path: aws_ofi_nccl_path.clone(),
                                        openmpi_path: openmpi_path.clone(),
                                        msccl_path: msccl_path.clone(),

                                        // Exe params
                                        executable: nccl_test_executable.clone(),

                                        // MSCCL params
                                        algorithm: comm_algorithm.to_string(),
                                        ms_xml_file: xml_file,
                                        ms_channels: msccl_channels.clone(),
                                        ms_chunks: msccl_chunks.clone(),
                                        gpu_as_node,
                                        num_nodes,
                                        total_gpus: num_gpus,
                                        buffer_size,

                                        // MPI Params
                                        mpi_hostfile_path: mpi_hostfile_path.clone(),
                                        mpi_proc_per_node: gpus_per_node.clone(),

                                        // NCCL Tests params
                                        nc_collective: collective.to_string(),
                                        nc_op: reduction_op.to_string(),
                                        nc_dtype: data_type.to_string(),
                                        nc_num_threads: 1,
                                        nc_num_gpus: 1,
                                        nc_min_bytes: message_size_range.0.to_string(),
                                        nc_max_bytes: message_size_range.1.to_string(),
                                        nc_step_factor: "2".to_string(),
                                        nc_num_iters: 100,
                                        nc_num_warmup_iters: 20,

                                        // NCCL Env params
                                        nccl_debug_level: nccl_debug_level.to_string(),
                                        nccl_algo:
                                            "Tree,Ring,CollnetDirect,CollnetChain,NVLS,NVLSTree"
                                                .to_string(), // Default NCCL
                                    };

                                    // Add the full experiment to the list
                                    experiment_descriptors.push(experiment);

                                    // Add the permutation to the list
                                    permutations.push(Permutation {
                                        collective_exe: collective_exe.to_string(),
                                        data_type: data_type.to_string(),
                                        reduction_op: reduction_op.to_string(),
                                        comm_algorithm: comm_algorithm.to_string(),
                                        msccl_channel: Some(msccl_channels.to_string()),
                                        msccl_chunk: Some(msccl_chunks.to_string()),
                                        buffer_size: Some(buffer_size.to_string()),
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    debug!("Finished generating all permutations/experiment configs.");

    // Pretty-print the permutations
    pretty_print_configs(&experiment_descriptors, false);

    // Create the record-keeping manifest
    let mut manifest_collection = Vec::new();

    // ACTUALLY run experiments by iterating over the list of permutations
    let total_experiments = experiment_descriptors.len() * num_repetitions;
    for (progress, experiment_descriptor) in experiment_descriptors.iter().enumerate() {
        for i in 0..num_repetitions {
            // debug!("Experiment descriptor found: {:#?}", experiment_descriptor);

            // Print info about this experiment
            // info!("Running collective {} (Op: {}) with data type: {}, comm algorithm: {}, MSCCL channel: {}, MSCCL chunk: {} ({} of {})",
            //     collective_exe, reduction_op, data_type, comm_algorithm, msccl_channel, msccl_chunk, i + 1, num_repetitions);
            info!(
                "### Running experiment [ # nodes: {} | # GPUs: {} | collective: {} | op: {} | dtype: {} | algorithm: {} | channels: {} | chunks: {} | buffer size: {} | GPU as Node: {:#?} | experiment {} of {} ] ###",
                experiment_descriptor.num_nodes,
                experiment_descriptor.total_gpus,
                experiment_descriptor.nc_collective,
                experiment_descriptor.nc_op,
                experiment_descriptor.nc_dtype,
                experiment_descriptor.algorithm,
                experiment_descriptor.ms_channels,
                experiment_descriptor.ms_chunks,
                experiment_descriptor.buffer_size,
                experiment_descriptor.gpu_as_node,
                i + 1,
                num_repetitions
            );

            info!(
                "Will attempt to use MSCCL XML file at: {}",
                experiment_descriptor.ms_xml_file.to_str().unwrap()
            );

            // Get the output file paths
            let output_path = experiments_output_dir.clone().join(
                exp_params_to_output_filename(&experiment_descriptor, i as u64, "log"),
            );
            let stderr_path = experiments_output_dir.clone().join(
                exp_params_to_output_filename(&experiment_descriptor, i as u64, "stderr")
            );

            // Skip blacklisted XML files
            for blacklisted in blacklist.iter() {
                let full_blacklisted_path = msccl_xmls_directory.join(blacklisted);

                if !full_blacklisted_path.exists() {
                    warn!("Blacklisted XML file not found at: {}. Skipping, but this is probably a bug in nccl_harness!", 
                        full_blacklisted_path.to_str().unwrap());
                }

                if experiment_descriptor.ms_xml_file == full_blacklisted_path {
                    info!("Skipping experiment because XML file is blacklisted: {:?}", experiment_descriptor.ms_xml_file);

                    // Update manifest
                    manifest_collection.push(ManifestEntry {
                        collective: experiment_descriptor.nc_collective.clone(),
                        op: experiment_descriptor.nc_op.clone(),
                        dtype: experiment_descriptor.nc_dtype.clone(),
                        algorithm: experiment_descriptor.algorithm.clone(),
                        num_channels: experiment_descriptor.ms_channels,
                        num_chunks: experiment_descriptor.ms_chunks,
                        num_gpus: experiment_descriptor.total_gpus,
                        buffer_size_factor: experiment_descriptor.buffer_size,
                        overall_result: ResultDescription::Blacklisted,
                    });

                    info!("---------------------------------------");

                    continue;
                }
            }

            // Skip if already completed and skip envvar is set
            if skip_finished && output_path.exists() {
                info!("Skipping experiment because output file already exists at: {:?} and 'SKIP_COMPLETED' envvar is set.", output_path);

                // Update manifest
                manifest_collection.push(ManifestEntry {
                    collective: experiment_descriptor.nc_collective.clone(),
                    op: experiment_descriptor.nc_op.clone(),
                    dtype: experiment_descriptor.nc_dtype.clone(),
                    algorithm: experiment_descriptor.algorithm.clone(),
                    num_channels: experiment_descriptor.ms_channels,
                    num_chunks: experiment_descriptor.ms_chunks,
                    num_gpus: experiment_descriptor.total_gpus,
                    buffer_size_factor: experiment_descriptor.buffer_size,
                    overall_result: ResultDescription::Skipped,
                });

                info!("---------------------------------------");

                continue;
            }

            let rows = match run_msccl_tests(
                &experiment_descriptor.executable,
                &experiment_descriptor,
                true, // Why? Well, Liuyao's testo sometimes return a nonzero status code
                dry_run,
                Some(output_path.clone()),
                Some(stderr_path.clone()),
            ) {
                Ok(v) => v,
                Err(e) => {
                    error!(
                        "Encountered an error while running NCCL Tests: {}. Continuing...",
                        e
                    );

                    // Update manifest
                    manifest_collection.push(ManifestEntry {
                        collective: experiment_descriptor.nc_collective.clone(),
                        op: experiment_descriptor.nc_op.clone(),
                        dtype: experiment_descriptor.nc_dtype.clone(),
                        algorithm: experiment_descriptor.algorithm.clone(),
                        num_channels: experiment_descriptor.ms_channels,
                        num_chunks: experiment_descriptor.ms_chunks,
                        num_gpus: experiment_descriptor.total_gpus,
                        buffer_size_factor: experiment_descriptor.buffer_size,
                        overall_result: ResultDescription::Failure,
                    });

                    info!("---------------------------------------");

                    // Continue to next experiments
                    continue;
                }
            };

            info!(
                "Finished running experiment. Completed {} of {} experiments ({:.1}%).",
                progress * 2 + i + 1,
                total_experiments,
                if total_experiments > 0 {
                    ((progress * 2 + i + 1) as f64 / total_experiments as f64) * 100.0
                } else {
                    100.0
                }
            );

            // Update manifest
            manifest_collection.push(ManifestEntry {
                collective: experiment_descriptor.nc_collective.clone(),
                op: experiment_descriptor.nc_op.clone(),
                dtype: experiment_descriptor.nc_dtype.clone(),
                algorithm: experiment_descriptor.algorithm.clone(),
                num_channels: experiment_descriptor.ms_channels,
                num_chunks: experiment_descriptor.ms_chunks,
                num_gpus: experiment_descriptor.total_gpus,
                buffer_size_factor: experiment_descriptor.buffer_size,
                overall_result: ResultDescription::Success,
            });

            // Print line separator
            info!("---------------------------------------");
        }
    }

    // Pretty Print the Manifest
    println!("\n\n\n--- 📋📋📋 EXPERIMENT RESULTS 📋📋📋 ---\n");
    pretty_print_result_manifest(&manifest_collection);

    Ok(())
}

// /// Run NCCL tests with MPI using a set of parameters
// fn run_nccl_test(hostfile_path: &Path, executable: &Path, msccl_xml_file: Option<&Path>,
//     proc_per_node: &str, num_threads: &str, num_gpus: &str, min_bytes: &str, max_bytes: &str, step_factor: &str, 
//     op: &str, datatype: &str, num_iters: &str, num_warmup_iters: &str, nccl_debug_level: &str, ignore_error_status_codes: bool) -> Result<Vec<Row>, Box<dyn std::error::Error>> {

//     // // Open output files
//     // let stdout_file = match stdout_output.exists() {
//     //     true => std::fs::OpenOptions::new().append(true).open(stdout_output)?,
//     //     false => std::fs::File::create(stdout_output)?
//     // };
//     // let stderr_file = match stderr_output.exists() {
//     //     true => std::fs::OpenOptions::new().append(true).open(stderr_output)?,
//     //     false => std::fs::File::create(stderr_output)?
//     // };
//     // let mut stdout_writer = std::io::BufWriter::new(stdout_file);
//     // let mut stderr_writer = std::io::BufWriter::new(stderr_file);

//     // MSCCL XML file handling (just use dummy envvar if not given an XML file)
//     let msccl_xml_envvar = match msccl_xml_file {
//         Some(p) => {
//             println!("Using MSCCL XML file at: {}", p.to_str().unwrap());
//             format!("MSCCL_XML_FILES={}", p.to_str().unwrap())
//         },
//         None => {
//             println!("[INFO] No MSCCL XML file was given, so using dummy envvar.");
//             "DUMMY_VAR=TRUE".to_string()
//         }
//     };

//     // Other MSCCL envvar
//     let gen_msccl_xml = match msccl_xml_file {
//         Some(_) => "GENMSCCLXML=1".to_string(),
//         None => "GDUMMY_VAR=TRUE".to_string()
//     };

//     // Run NCCL tests with MPI
//     // TODO: Verify that OpenMPI passes through required environment variables
//     println!("Running NCCL tests with MPI...");
//     let mut res = Command::new("mpirun")
//         .args(["--hostfile", hostfile_path.to_str().unwrap()])
//         .args(["--map-by", format!("ppr:{}:node", proc_per_node).as_str()])
//         // [HACK] FIXME: This hardcoded LD_LIBRARY_PATH is awful and needs to be removed!
//         .args(["-x", format!("LD_LIBRARY_PATH=/opt/aws-ofi-nccl-lyd/lib:/opt/amazon/openmpi/lib64:/home/ec2-user/deps/msccl/build/lib:/usr/local/cuda/lib64:{}", std::env::var("LD_LIBRARY_PATH").unwrap().as_str()).as_str()])
//         .args(["-x", msccl_xml_envvar.as_str()])
//         .args(["-x", gen_msccl_xml.as_str()])
//         .args(["-x", format!("NCCL_DEBUG={}", nccl_debug_level).as_str()])
//         .args(["-x", "FI_EFA_USE_DEVICE_RDMA=1"])
//         .args(["-x", "FI_EFA_FORK_SAFE=1"])
//         .args(["--mca", "btl", "tcp,self", "--mca", "btl_tcp_if_exclude", "lo,docker0", "--bind-to", "none"])
//         .arg(executable.to_str().unwrap())
//         .args(["--nthreads", format!("{}", num_threads).as_str()])
//         .args(["--ngpus", num_gpus])
//         .args(["--minbytes", min_bytes])
//         .args(["--maxbytes", max_bytes])
//         .args(["--stepfactor", step_factor])
//         .args(["--op", op])
//         .args(["--datatype", datatype])
//         .args(["--iters", num_iters])
//         .args(["--warmup_iters", num_warmup_iters])
//         .stdout(std::process::Stdio::piped())
//         .stderr(std::process::Stdio::piped())
//         .spawn()
//         .expect("[ERROR] FAILED TO RUN WITH MPI!!!!");

//     // Create vector to store rows
//     let mut rows = Vec::new();

//     // Print and handle stdout line by line
//     let stdout_reader = std::io::BufReader::new(res.stdout.take().unwrap());
//     // let reader = std::io::BufReader::new(res.stdout.take().unwrap().as_fd());
//     for line in stdout_reader.lines() {
//         match line {
//             Ok(line) => {
//                 // // Write line to file
//                 // match stdout_writer.write_all(line.as_bytes()) {
//                 //     Ok(_) => {},
//                 //     Err(e) => {
//                 //         println!("[E]: Error writing line to stdout file: {}", e);
//                 //     }
//                 // }

//                 // Parse line, get row if this is a table data row
//                 if let Some(row) = parse_line(line.as_str()).unwrap() {
//                     rows.push(row);
//                     println!("[r]: {}", line);
//                 } 
                
//                 // Just print the line if it isn't a table data row
//                 else {
//                     println!("[l]: {}", line);
//                 }
//             },
//             Err(e) => {
//                 println!("Error parsing line: {}", e);
//             }
//         }
//     }

//     // Print stderr
//     // FIXME: Won't actually print if there's a hang-related error! The stdout reader never finishes reading!
//     let stderr_reader = std::io::BufReader::new(res.stderr.take().unwrap());
//     for line in stderr_reader.lines() {
//         match line {
//             Ok(line) => {
//                 // // Write line to file
//                 // match stderr_writer.write_all(line.as_bytes()) {
//                 //     Ok(_) => {},
//                 //     Err(e) => {
//                 //         println!("[E]: Error writing line to stdout file: {}", e);
//                 //     }
//                 // }

//                 // Print the line
//                 println!("[E]: {}", line);
//             },
//             Err(e) => {
//                 println!("[ERROR] Error getting line from stdout: {}", e);
//             }
//         }
//     }

//     // Handle exit status
//     let status = res.wait()?;
//     match status.success() {
//         true => println!("NCCL tests with MPI ran successfully."),
//         false => {
//             if !ignore_error_status_codes {
//                 println!("NCCL tests with MPI failed with exit code: {}", status.code().unwrap());
//                 return Err("NCCL tests with MPI failed.".into());
//             } else {
//                 println!("NCCL tests with MPI failed with exit code: {}, but ignoring and continuing.", 
//                     status.code().unwrap());
//             }
//         }
//     }

//     Ok(rows)
// }

// /// Struct that describes a set of parameters to run MSCCL with
// struct MscclExperimentParams {
//     // Environment Params
//     cuda_path: &str,
//     efa_path: Option<&str>,
//     aws_ofi_nccl_path: Option<&str>,
//     openmpi_path: &str,
//     msccl_path: &str,

//     // MSCCL Params
//     xml_file: &Path,

//     // MPI Params
//     hostfile_path: &Path,
//     mpi_proc_per_node: u64,

//     // NCCL Tests Params
//     nc_op: &str,
//     nc_dtype: &str,
//     nc_num_threads: u64,
//     nc_num_gpus: u64,
//     nc_min_bytes: &str,
//     nc_max_bytes: &str,
//     nc_step_factor: &str,
//     nc_num_iters: u64,
//     nc_num_warmup_iters: u64,

//     // NCCL Env Params
//     nccl_debug_level: &str,
//     nccl_algo: &str,
// }
