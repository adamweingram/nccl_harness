use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::Command;
use regex::Regex;
use polars::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {

    // NCCL tests executable binary location
    let nccl_test_bins = Path::new(
        "/home/Adam/Projects/NCCL-Harness/nccl-experiments/nccl-tests/build"
    );
    assert!(nccl_test_bins.exists());

    // Experimental setup
    let num_repetitions = 2;
    // let data_types = ["double", "float", "int32", "int8"];
    let data_types = ["float", "int32"];
    // let collective_exes = ["all_reduce_perf", "all_gather_perf", "alltoall_perf", "broadcast_perf", "gather_perf", "hypercube_perf", "reduce_perf", "reduce_scatter_perf", "scatter_perf", "sendrecv_perf"];
    let collective_exes = ["all_reduce_perf", "all_gather_perf"];
    let nccl_debug_level = "INFO";  // Use `TRACE` for replayable trace information on every call

    // Run experiments
    for collective_exe in collective_exes {

        // Build executable path
        let nccl_test_executable = nccl_test_bins.join(collective_exe);
        assert!(nccl_test_executable.exists());

        for data_type in data_types {
            for i in 0..num_repetitions {
                println!("Running of collective {} with data type: {}, ({} of {})", collective_exe, data_type, i + 1, num_repetitions);
                run_nccl_test(
                    &nccl_test_executable,
                    "1", 
                    "1", 
                    "1", 
                    "2", 
                    "512M", 
                    "2", 
                    "sum", 
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
                println!("---------------------------------");
            }
        }
    }

    Ok(())
}

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
    oop_num_wrong: u64,
    ip_time: f64,
    ip_alg_bw: f64,
    ip_bus_bw: f64,
    ip_num_wrong: u64
}

fn parse_line(line: &str) -> Result<Option<Row>, Box<dyn std::error::Error>> {
    let line_slice = line.split_whitespace().collect::<Vec<&str>>();

    // Describes the prelude to a logfile
    let re = Regex::new(r"[A-z0-9]+:[0-9]+:[0-9]+").unwrap();

    // Handle log rows
    if re.is_match(line) {
        println!("[l]: {:?}", line);
        return Ok(None);
    } 
    
    // Handle table data rows
    else if line_slice.len() == 13 {
        // 13 columns in the NCCL output table
        // println!("Data Slice: {:?}", line_slice);
        
        // Create row
        let row = Row {
            size: line_slice[0].parse::<u64>().unwrap(),
            count: line_slice[1].parse::<u64>().unwrap(),
            dtype: line_slice[2].to_string(),
            redop: line_slice[3].to_string(),
            root: line_slice[4].parse::<i64>().unwrap(),
            oop_time: line_slice[5].parse::<f64>().unwrap(),
            oop_alg_bw: line_slice[6].parse::<f64>().unwrap(),
            oop_bus_bw: line_slice[7].parse::<f64>().unwrap(),
            oop_num_wrong: line_slice[8].parse::<u64>().unwrap(),
            ip_time: line_slice[9].parse::<f64>().unwrap(),
            ip_alg_bw: line_slice[10].parse::<f64>().unwrap(),
            ip_bus_bw: line_slice[11].parse::<f64>().unwrap(),
            ip_num_wrong: line_slice[12].parse::<u64>().unwrap()
        };
        println!("Row: {:?}", row);

        // Return that a line was successfully parsed
        return Ok(Some(row));
    }

    Ok(None)
}

fn run_nccl_test(executable: &PathBuf, proc_per_node: &str, num_threads: &str, num_gpus: &str, min_bytes: &str, max_bytes: &str, 
    step_factor: &str, op: &str, datatype: &str, root: &str, num_iters: &str, num_warmup_iters: &str, agg_iters: &str,
    average: &str, parallel_init: &str, check: &str, blocking: &str, cuda_graph: &str, nccl_debug_level: &str) -> Result<(), Box<dyn std::error::Error>> {

    // Run NCCL tests with MPI
    let mut res = Command::new("mpirun")
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
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("Failed to run with MPI.");

    // Print full command with arguments
    println!("Command: {:?}", res);

    // Create dataframe to store results
    // let mut df = DataFrame::new(vec![
    //     Series::new("size", &[]),
    //     Series::new("time", &[]),
    //     Series::new("min", &[]),
    //     Series::new("max", &[]),
    //     Series::new("avg", &[]),
    //     Series::new("stddev", &[]),
    //     Series::new("count", &[]),
    //     Series::new("size", &[]),
    //     Series::new("time", &[]),
    //     Series::new("min", &[]),
    //     Series::new("max", &[]),
    //     Series::new("avg", &[]),
    //     Series::new("stddev", &[]),
    //     Series::new("count", &[]),
    // ]).unwrap();
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
                }

            },
            Err(e) => {
                println!("Error: {}", e);
            }
        }
    }
    println!("Finished running experiments.");
    println!("Rows: {:?}", rows);

    Ok(())
}

// fn run_nccl_tests(proc_per_node: u8) -> Result<(), Box<dyn std::error::Error>> {
//     let nccl_test_executable = Path::new(
//         "/home/Adam/Projects/NCCL-Harness/nccl-experiments/nccl-tests/build/all_reduce_perf"
//     );

//     assert!(nccl_test_executable.exists());

//     println!("Env: {:?}", std::env::vars());

//     // Experiment Parameters
//     let num_threads = "1";      // Number of threads per process. Default: 1.
//     let ngpus = "1";            // Number of gpus per thread. Default: 1.

//     let minbytes = "2";         // Minimum size to start with. Default: 32M.
//     let maxbytes = "13746984277";   // Maximum size to end at. Default: 32M. (NOTE: Automatically limited in certain situations.)
//     let stepfactor = "2";       // Multiplication factor between sizes. Default: disabled.

//     let op = "sum";          // Specify which reduction operation to perform. Only relevant for reduction operations like Allreduce, Reduce or ReduceScatter. Default: Sum.
//     let datatype = "float";  // Specify which datatype to use. Default: Float.
//     let root = "0";             // Specify which root to use. Only for operations with a root like broadcast or reduce. Default: 0.

//     let num_iters = "20";       // Number of iterations to run. Default: 20.
//     let num_warmup_iters = "5"; // Number of warmup iterations to run. Default: 5.
//     let agg_iters = "1";        // Number of iterations to aggregate. Default: 1.
//     let average = "1";          // Report performance as an average across all ranks (MPI=1 only). <0=Rank0,1=Avg,2=Min,3=Max>. Default : 1.

//     let parallel_init = "0"; // Use threads to initialize NCCL in parallel. Default: 0.
//     let check = "1";         // Perform count iterations, checking correctness of results on each iteration. This can be quite slow on large numbers of GPUs. Default : 1.
//     let blocking = "0";      // Make NCCL collective blocking, i.e. have CPUs wait and sync after each collective. Default : 0.
//     let cuda_graph = "0";    // Capture iterations as a CUDA graph and then replay specified number of times. Default : 0.

//     // Run NCCL tests with MPI
//     let mut res = Command::new("mpirun")
//         .args(["--map-by", format!("ppr:{}:node", proc_per_node).as_str()])
//         .arg(nccl_test_executable.to_str().unwrap())
//         .args(["--nthreads", format!("{}", num_threads).as_str()])
//         .args(["--ngpus", ngpus])
//         .args(["--minbytes", minbytes])
//         .args(["--maxbytes", maxbytes])
//         .args(["--stepfactor", stepfactor])
//         .args(["--op", op])
//         .args(["--datatype", datatype])
//         .args(["--root", root])
//         .args(["--iters", num_iters])
//         .args(["--warmup_iters", num_warmup_iters])
//         .args(["--agg_iters", agg_iters])
//         .args(["--average", average])
//         .args(["--parallel_init", parallel_init])
//         .args(["--check", check])
//         .args(["--blocking", blocking])
//         .args(["--cudagraph", cuda_graph])
//         .stdout(std::process::Stdio::piped())
//         .stderr(std::process::Stdio::piped())
//         .spawn()
//         .expect("Failed to run with MPI.");

//     // Print full command with arguments
//     println!("Command: {:?}", res);

//     // Print output
//     let reader = std::io::BufReader::new(res.stdout.take().unwrap());
//     // let reader = std::io::BufReader::new(res.stdout.take().unwrap().as_fd());
//     for line in reader.lines() {
//         println!("Found {}", line.unwrap());
//     }
//     println!("Finished running experiments.");

//     // let mut res = Command::new("mpirun")
//     //     .arg("--map-by")
//     //     .arg(format!("ppr:{}:node", proc_per_node))
//     //     .arg(nccl_test_executable.to_str().unwrap())
//     //     .arg("-b")
//     //     .arg("8")
//     //     .arg("-e")
//     //     .arg("512M")
//     //     .arg("-f")
//     //     .arg("2")
//     //     .arg("-g")
//     //     .arg("1")
//     //     .stdout(std::process::Stdio::inherit())
//     //     .stderr(std::process::Stdio::inherit())
//     //     .spawn()
//     //     .expect("Failed to run with MPI.");
//     // res.wait().unwrap();

//     // println!("Full Result: {}", str::from_utf8(res.stdout.as_slice()).unwrap());

//     Ok(())
// }
