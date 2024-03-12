use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::Command;
use regex::Regex;
use polars::prelude::*;

/// Struct to describe a table row from the NCCL output
#[derive(Debug, Clone)]
struct Row {
    size: u64,
    count: u64,
    dtype: String,
    redop: String,
    root: i64,
    oop_time: f64,
    oop_alg_bw: f64,
    oop_bus_bw: f64,
    oop_num_wrong: String,  // Sometimes is N/A, so can't use u64
    ip_time: f64,
    ip_alg_bw: f64,
    ip_bus_bw: f64,
    ip_num_wrong: String    // Sometimes is N/A, so can't use u64
}

fn verify_env() -> Result<(), Box<dyn std::error::Error>> {
    // Verify environment variables are set and paths are accessible
    let nccl_home = PathBuf::from(std::env::var("NCCL_HOME").expect("[ERROR] NCCL_HOME not set!"));
    let cuda_home = PathBuf::from(std::env::var("CUDA_HOME").expect("[ERROR] CUDA_HOME not set!"));
    let mpi_home = PathBuf::from(std::env::var("MPI_HOME").expect("[ERROR] MPI_HOME not set!"));
    let nccl_tests_home = PathBuf::from(std::env::var("NCCL_TESTS_HOME").expect("[ERROR] NCCL_TESTS_HOME not set!"));
    let _ = PathBuf::from(std::env::var("EXPERIMENTS_OUTPUT_DIR").expect("[ERROR] EXPERIMENTS_OUTPUT_DIR not set!"));
    let mpi_hostfile = PathBuf::from(std::env::var("MPI_HOSTFILE").expect("[ERROR] MPI_HOSTFILE not set!"));
    let msccl_xmls = PathBuf::from(std::env::var("MSCCL_XMLS").expect("[ERROR] MSCCL_XMLS not set!"));
    if !nccl_home.exists() {
        panic!("[ERROR] NCCL_HOME not found at: {}", nccl_home.to_str().unwrap());
    }
    if !cuda_home.exists() {
        panic!("[ERROR] CUDA_HOME not found at: {}", cuda_home.to_str().unwrap());
    }
    if !mpi_home.exists() {
        panic!("[ERROR] MPI_HOME not found at: {}", mpi_home.to_str().unwrap());
    }
    if !nccl_tests_home.exists() {
        panic!("[ERROR] NCCL_TESTS_HOME not found at: {}", nccl_tests_home.to_str().unwrap());
    }
    // // Note: We don't need this to be created as it will be created if it doesn't exist
    // if !experiments_output.exists() {
    //     panic!("[ERROR] EXPERIMENTS_OUTPUT_DIR not found at: {}", experiments_output.to_str().unwrap());
    // }
    if !mpi_hostfile.exists() {
        panic!("[ERROR] MPI_HOSTFILE not found at: {}", mpi_hostfile.to_str().unwrap());
    }
    if !msccl_xmls.exists() {
        panic!("[ERROR] MSCCL_XMLS not found at: {}", msccl_xmls.to_str().unwrap());
    }

    let nccl_lib = nccl_home.join("lib");
    let cuda_lib = cuda_home.join("lib64");
    let mpi_lib = mpi_home.join("lib64");
    if !nccl_lib.exists() {
        panic!("[ERROR] NCCL lib not found at: {}", nccl_lib.to_str().unwrap());
    }
    if !cuda_lib.exists() {
        panic!("[ERROR] CUDA lib not found at: {}", cuda_lib.to_str().unwrap());
    }
    if !mpi_lib.exists() {
        panic!("[ERROR] MPI lib not found at: {}", mpi_lib.to_str().unwrap());
    }

    // let path = std::env::var("PATH").unwrap();
    let ld_library_path = std::env::var("LD_LIBRARY_PATH").unwrap();

    // Verify that the necessary libraries are in the LD_LIBRARY_PATH
    for lib in [nccl_lib, cuda_lib, mpi_lib].iter() {
        if !ld_library_path.contains(lib.to_str().unwrap()) {
            panic!("[ERROR] {} not in LD_LIBRARY_PATH!", lib.to_str().unwrap())
        }

        if !ld_library_path.contains(lib.to_str().unwrap()) {
            panic!("[ERROR] {} not in LD_LIBRARY_PATH!", lib.to_str().unwrap())
        }
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Verify environment variables
    verify_env()?;

    // MPI hostfile
    let mpi_hostfile = PathBuf::from(std::env::var("MPI_HOSTFILE").unwrap());
    if !mpi_hostfile.exists() {
        panic!("[ERROR] MPI_HOSTFILE not found at: {}", mpi_hostfile.to_str().unwrap());
    }

    // Output data directory
    let experiments_output_dir = PathBuf::from(std::env::var("EXPERIMENTS_OUTPUT_DIR").unwrap());
    if !experiments_output_dir.exists() {
        std::fs::create_dir(experiments_output_dir.as_path())?;
    }

    // NCCL tests executable binary location
    let nccl_test_bins = PathBuf::from(std::env::var("NCCL_TESTS_HOME").unwrap());

    // MSCCL XML files location
    let msccl_xmls_directory = PathBuf::from(std::env::var("MSCCL_XMLS").unwrap());

    // Experimental setup
    let num_repetitions = 2;
    let data_types = [
        // "double", 
        "float", 
        // "int32", 
        "int8"
    ];
    let collective_exes = [
        "all_reduce_perf", 
        // "all_gather_perf", 
        // "alltoall_perf", 
        // "broadcast_perf", 
        // "gather_perf", 
        // "hypercube_perf",  // BROKEN FOR HYPERCUBE BECAUSE THE OUTPUT TABLE IS BLANK FOR REDOP (breaks parsing)
        // "reduce_perf", 
        // "reduce_scatter_perf", 
        // "scatter_perf", 
        // "sendrecv_perf"
    ];
    let reduction_ops = [
        "sum", 
        // "prod", 
        // "min", 
        // "max",
        // "avg"
    ];
    let msccl_channels = [
        "1",
        // "2" // Probably broken because of single NIC
    ];
    let msccl_chunks = [
        "1",
        "2",
        "4",
        "8",
        "16",
        "32",
        "64"
    ];
    let comm_algorithms = [
        "binary_tree",
        "binomial_tree",
        "recursive_doubling",
        "recursive_doubling_halving",
        "ring",
        "triple_trinomial_tree"
    ];

    let nccl_debug_level = "INFO";  // Use `TRACE` for replayable trace information on every call

    // Run experiments
    for collective_exe in collective_exes {

        // Build executable path
        let nccl_test_executable = nccl_test_bins.join(collective_exe);
        assert!(nccl_test_executable.exists());

        // Run experiments across all variations
        for data_type in data_types {
            for reduction_op in reduction_ops {
                for comm_algorithm in comm_algorithms {
                    for msccl_channel in msccl_channels {
                        for msccl_chunk in msccl_chunks {
                            for i in 0..num_repetitions {
                                // Print info about this experiment
                                println!("Running collective {} (Op: {}) with data type: {}, comm algorithm: {}, MSCCL channel: {}, MSCCL chunk: {} ({} of {})", 
                                    collective_exe, reduction_op, data_type, comm_algorithm, msccl_channel, msccl_chunk, i + 1, num_repetitions);

                                // Handle non-created XMLs
                                // FIXME: Ugly hack
                                if collective_exe == "all_reduce_perf" && 
                                    comm_algorithm == "ring" &&
                                    (msccl_chunk != "1" || 
                                     msccl_channel != "2" || 
                                     msccl_channel != "4" ||
                                     msccl_channel != "8")
                                     {
                                    println!("[INFO] Skipping ring algorithm for all_reduce_perf because it is not supported by Liuyao's generated XML! (Chunk was {}, Channel was {})", 
                                        msccl_chunk, 
                                        msccl_channel);

                                    // Skip this iteration and therefore experiment
                                    continue;
                                }

                                // Find name of the collective algorithm (different from the binary)
                                let algo_name = match collective_exe {
                                    "all_reduce_perf" => "allreduce",
                                    "all_gather_perf" => "allgather",
                                    "alltoall_perf" => "alltoall",
                                    "broadcast_perf" => "broadcast",
                                    "gather_perf" => "gather",
                                    "hypercube_perf" => "hypercube",
                                    "reduce_perf" => "reduce",
                                    "reduce_scatter_perf" => "reducescatter",
                                    "scatter_perf" => "scatter",
                                    "sendrecv_perf" => "sendrecv",
                                    _ => panic!("[ERROR] Unknown collective algorithm: {}", collective_exe)
                                };

                                // Select correct XML file
                                let xml_file = msccl_xmls_directory.join(format!(
                                    "{}_{}_{}ch_{}chunk.xml", 
                                    algo_name, 
                                    comm_algorithm,
                                    msccl_channel,
                                    msccl_chunk));
                                
                                if !xml_file.exists() {
                                    println!("[ERROR] XML file not found at: {}", xml_file.to_str().unwrap());
                                    continue;
                                }
                                println!("Will attempt to use MSCCL XML file at: {}", xml_file.to_str().unwrap());

                                // Run NCCL test
                                println!("Running of collective {} (Op: {}) with data type: {}, ({} of {})", collective_exe, reduction_op, data_type, i + 1, num_repetitions);
                                let rows = run_nccl_test(
                                    &mpi_hostfile,
                                    &nccl_test_executable,
                                    &xml_file,
                                    "8", // 8xA100 GPUs per node
                                    "1", 
                                    "1",      // 1 GPU per MPI process
                                    "2", 
                                    "512M", 
                                    "2", 
                                    reduction_op, 
                                    data_type, 
                                    "0", 
                                    "20", 
                                    "5", 
                                    "1", 
                                    "1", 
                                    "0", 
                                    "1", 
                                    "0", 
                                    "0",
                                    nccl_debug_level).unwrap();

                                // Convert rows to DataFrame
                                let mut df = rows_to_df(rows).unwrap();
                                println!("DataFrame: {:?}", df);

                                // Write to CSV
                                let csv_file = experiments_output_dir.as_path().join(format!("{}_{}_{}_{}.csv", collective_exe, reduction_op, data_type, i));
                                println!("Writing results to CSV at {}...", csv_file.to_str().unwrap());
                                let opened_file = std::fs::File::create(&csv_file)?;
                                CsvWriter::new(opened_file)
                                    .finish(&mut df)?;
                                println!("Wrote results to CSV at {}.", csv_file.to_str().unwrap());

                                // Print line separator
                                println!("---------------------------------");
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Convert rows to a Polars DataFrame
/// 
/// Note: The implementaiton is very manual and not efficient.
fn rows_to_df(rows: Vec<Row>) -> Result<DataFrame, Box<dyn std::error::Error>> {
    // Create the dataframe
    let df = DataFrame::new(vec![
        Series::new("size", rows.iter().map(|r| r.size).collect::<Vec<u64>>()),
        Series::new("count", rows.iter().map(|r| r.count).collect::<Vec<u64>>()),
        Series::new("dtype", rows.iter().map(|r| r.dtype.clone()).collect::<Vec<String>>()),
        Series::new("redop", rows.iter().map(|r| r.redop.clone()).collect::<Vec<String>>()),
        Series::new("root", rows.iter().map(|r| r.root).collect::<Vec<i64>>()),
        Series::new("oop_time", rows.iter().map(|r| r.oop_time).collect::<Vec<f64>>()),
        Series::new("oop_alg_bw", rows.iter().map(|r| r.oop_alg_bw).collect::<Vec<f64>>()),
        Series::new("oop_bus_bw", rows.iter().map(|r| r.oop_bus_bw).collect::<Vec<f64>>()),
        Series::new("oop_num_wrong", rows.iter().map(|r| r.oop_num_wrong.clone()).collect::<Vec<String>>()),
        Series::new("ip_time", rows.iter().map(|r| r.ip_time).collect::<Vec<f64>>()),
        Series::new("ip_alg_bw", rows.iter().map(|r| r.ip_alg_bw).collect::<Vec<f64>>()),
        Series::new("ip_bus_bw", rows.iter().map(|r| r.ip_bus_bw).collect::<Vec<f64>>()),
        Series::new("ip_num_wrong", rows.iter().map(|r| r.ip_num_wrong.clone()).collect::<Vec<String>>())
    ])?;

    Ok(df)
}

/// Parse a line from the NCCL output
/// 
/// Note: Only returns something if the line is a table data row
fn parse_line(line: &str) -> Result<Option<Row>, Box<dyn std::error::Error>> {
    let line_slice = line.split_whitespace().collect::<Vec<&str>>();

    // Describes the prelude to a logfile
    let re = Regex::new(r"[A-z0-9]+:[0-9]+:[0-9]+").unwrap();

    // Handle log rows
    if re.is_match(line) {
        // println!("[l]: {:?}", line);
        return Ok(None);
    } 
    
    // Handle table data rows
    else if line_slice.len() == 13 {
        // 13 columns in the NCCL output table
        // println!("Data Slice: {:?}", line_slice);
        
        // Create row
        // TODO: Handle parse errors
        let row = Row {
            size: match line_slice[0].parse::<u64>() {
                Ok(v) => v,
                Err(e) => {
                    println!("Error parsing size: {}", e);
                    return Ok(None);
                }
            
            },
            count: match line_slice[1].parse::<u64>() {
                Ok(v) => v,
                Err(e) => {
                    println!("Error parsing count: {}", e);
                    return Ok(None);
                }
            },
            dtype: line_slice[2].to_string(),
            redop: match line_slice[3].to_string().is_empty() {
                true => "N/A".to_string(),
                false => line_slice[3].to_string()
            
            },
            root: match line_slice[4].parse::<i64>() {
                Ok(v) => v,
                Err(e) => {
                    println!("Error parsing root: {}", e);
                    return Ok(None);
                }
            },
            oop_time: match line_slice[5].parse::<f64>() {
                Ok(v) => v,
                Err(e) => {
                    println!("Error parsing oop_time: {}", e);
                    return Ok(None);
                }
            },
            oop_alg_bw: match line_slice[6].parse::<f64>() {
                Ok(v) => v,
                Err(e) => {
                    println!("Error parsing oop_alg_bw: {}", e);
                    return Ok(None);
                }
            },
            oop_bus_bw: match line_slice[7].parse::<f64>() {
                Ok(v) => v,
                Err(e) => {
                    println!("Error parsing oop_bus_bw: {}", e);
                    return Ok(None);
                }
            },
            oop_num_wrong: line_slice[8].to_string(),
            ip_time: match line_slice[9].parse::<f64>() {
                Ok(v) => v,
                Err(e) => {
                    println!("Error parsing ip_time: {}", e);
                    return Ok(None);
                }
            },
            ip_alg_bw: match line_slice[10].parse::<f64>() {
                Ok(v) => v,
                Err(e) => {
                    println!("Error parsing ip_alg_bw: {}", e);
                    return Ok(None);
                }
            },
            ip_bus_bw: match line_slice[11].parse::<f64>() {
                Ok(v) => v,
                Err(e) => {
                    println!("Error parsing ip_bus_bw: {}", e);
                    return Ok(None);
                }
            },
            ip_num_wrong: line_slice[12].to_string()
        };
        // println!("Row: {:?}", row);

        // Return that a line was successfully parsed
        return Ok(Some(row));
    }

    Ok(None)
}

/// Run NCCL tests with MPI using a set of parameters
fn run_nccl_test(hostfile_path: &Path, executable: &Path, msccl_xml_file: &Path, proc_per_node: &str, num_threads: &str, 
    num_gpus: &str, min_bytes: &str, max_bytes: &str, step_factor: &str, op: &str, datatype: &str, root: &str, 
    num_iters: &str, num_warmup_iters: &str, agg_iters: &str, average: &str, parallel_init: &str, check: &str, blocking: &str, 
    cuda_graph: &str, nccl_debug_level: &str) -> Result<Vec<Row>, Box<dyn std::error::Error>> {

    // Run NCCL tests with MPI
    // TODO: Verify that OpenMPI passes through required environment variables
    println!("Running NCCL tests with MPI...");
    let mut res = Command::new("mpirun")
        .args(["--hostfile", hostfile_path.to_str().unwrap()])
        .args(["--map-by", format!("ppr:{}:node", proc_per_node).as_str()])
        .arg(executable.to_str().unwrap())
        .args(["--nthreads", format!("{}", num_threads).as_str()])
        .args(["--ngpus", num_gpus])
        .args(["--minbytes", min_bytes])
        .args(["--maxbytes", max_bytes])
        .args(["--stepfactor", step_factor])
        .args(["--op", op])
        .args(["--datatype", datatype])
        .args(["--root", root])
        .args(["--iters", num_iters])
        .args(["--warmup_iters", num_warmup_iters])
        .args(["--agg_iters", agg_iters])
        .args(["--average", average])
        .args(["--parallel_init", parallel_init])
        .args(["--check", check])
        .args(["--blocking", blocking])
        .args(["--cudagraph", cuda_graph])
        .env("NCCL_DEBUG", nccl_debug_level)
        .env("MSCCL_XML_FILES", msccl_xml_file.to_str().unwrap())
        .env("FI_EFA_FORK_SAFE", "1")  // Necessary or OFI NCCL plugin crashes
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to run with MPI.");

    // Handle exit status
    let status = res.wait()?;

    // Create vector to store rows
    let mut rows = Vec::new();

    // Print output
    let reader = std::io::BufReader::new(res.stdout.take().unwrap());
    // let reader = std::io::BufReader::new(res.stdout.take().unwrap().as_fd());
    for line in reader.lines() {
        match line {
            Ok(line) => {
                // Parse line, get row if this is a table data row
                if let Some(row) = parse_line(line.as_str()).unwrap() {
                    rows.push(row);
                    println!("[r]: {}", line);
                } 
                
                // Just print the line if it isn't a table data row
                else {
                    println!("[l]: {}", line);
                }
            },
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }

    Ok(rows)
}
