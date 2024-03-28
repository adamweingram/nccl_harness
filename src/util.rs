use std::{fmt, path::{Path, PathBuf}};
use termion::color;

/// Struct to describe a table row from the NCCL output
#[derive(Debug, Clone)]
pub struct Row {
    pub size: u64,
    pub count: u64,
    pub dtype: String,
    pub redop: String,
    pub root: i64,
    pub oop_time: f64,
    pub oop_alg_bw: f64,
    pub oop_bus_bw: f64,
    pub oop_num_wrong: String, // Sometimes is N/A, so can't use u64
    pub ip_time: f64,
    pub ip_alg_bw: f64,
    pub ip_bus_bw: f64,
    pub ip_num_wrong: String, // Sometimes is N/A, so can't use u64
}

#[derive(Debug, Clone)]
pub struct Permutation {
    pub collective_exe: String,
    pub reduction_op: String,
    pub data_type: String,
    pub comm_algorithm: String,
    pub msccl_channel: Option<String>,
    pub msccl_chunk: Option<String>,
    pub buffer_size: Option<String>,
}

/// Struct that describes a set of parameters to run MSCCL with
#[derive(Debug, Clone)]
pub struct MscclExperimentParams {
    // Environment Params
    pub cuda_path: String,
    pub efa_path: Option<String>,
    pub aws_ofi_nccl_path: Option<String>,
    pub openmpi_path: String,
    pub msccl_path: String,

    // Exe params
    pub executable: PathBuf,

    // MSCCL Params
    pub algorithm: String,
    pub ms_xml_file: PathBuf,
    pub ms_channels: u64,
    pub ms_chunks: u64,
    pub gpu_as_node: bool,
    pub num_nodes: u64,
    pub total_gpus: u64,
    pub buffer_size: u64,

    // MPI Params
    pub mpi_hostfile_path: PathBuf,
    pub mpi_proc_per_node: u64,

    // NCCL Tests Params
    pub nc_collective: String,
    pub nc_op: String,
    pub nc_dtype: String,
    pub nc_num_threads: u64,
    pub nc_num_gpus: u64,
    pub nc_min_bytes: String,
    pub nc_max_bytes: String,
    pub nc_step_factor: String,
    pub nc_num_iters: u64,
    pub nc_num_warmup_iters: u64,

    // NCCL Env Params
    pub nccl_debug_level: String,
    pub nccl_algo: String,
}

/// Describes the result of an experiment
#[derive(Debug, Clone)]
pub enum ResultDescription {
    Success,
    PartialFailure,
    Failure,
    Skipped,
    Blacklisted,
}

impl fmt::Display for ResultDescription {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            ResultDescription::Success => write!(f, "Success"),
            ResultDescription::PartialFailure => write!(f, "Partial Failure"),
            ResultDescription::Failure => write!(f, "Failure"),
            ResultDescription::Skipped => write!(f, "Skipped"),
            ResultDescription::Blacklisted => write!(f, "Blacklisted"),
        }
    }
}

/// Struct the basic params and results of an experiment
#[derive(Debug, Clone)]
pub struct ManifestEntry {
    pub collective: String,
    pub op: String,
    pub dtype: String,
    pub algorithm: String,
    pub num_channels: u64,
    pub num_chunks: u64,
    pub num_gpus: u64,
    pub buffer_size_factor: u64,

    pub overall_result: ResultDescription,
}

/// Get the name of the output file for a set of given MSCCL experiment parameters
#[inline(always)]
pub fn exp_params_to_output_filename(params: &MscclExperimentParams, iteration: u64, extension: &str) -> PathBuf {
    // (collective)_(algorithm)_node(# nodes)_gpu(# gpus)_mcl(# channels)_mck(# chunks)_buf(scl. fac.)_gan(1|0)_i(iter id).(extension)
    PathBuf::from(format!(
        "{}_{}_node{}_gpu{}_mcl{}_mck{}_buf{}_gan{}_i{}.{}",
        params.nc_collective,
        params.algorithm,
        params.num_nodes,
        params.total_gpus,
        params.ms_channels,
        params.ms_chunks,
        params.buffer_size,
        if params.gpu_as_node { 1 } else { 0 },
        iteration,
        extension,
    ))
}

/// Get the name of the NCCL-tests executable that corresponds to the given collective name.
/// 
/// # Arguments
/// * `collective` - The name of the collective to get the corresponding NCCL-tests executable name for
/// 
/// # Returns
/// The name of the NCCL-tests executable that corresponds to the given collective name
#[inline(always)]
pub fn collective_to_test_exe(collective: &str) -> Result<String, Box<dyn std::error::Error>> {
    match collective {
        "all-reduce" => Ok("all_reduce_perf".to_string()),
        "all-gather" => Ok("all_gather_perf".to_string()),
        "all-to-all" => Ok("alltoall_perf".to_string()),
        "broadcast" => Ok("broadcast_perf".to_string()),
        "gather" => Ok("gather_perf".to_string()),
        "hypercube" => Ok("hypercube_perf".to_string()),
        "reduce" => Ok("reduce_perf".to_string()),
        "reduce-scatter" => Ok("reduce_scatter_perf".to_string()),
        "scatter" => Ok("scatter_perf".to_string()),
        "sendrecv" => Ok("sendrecv_perf".to_string()),
        _ => {
            return Err(format!("Could not figure out which NCCL-tests executable this collective name this corresponds to: {}", collective).into());
        }
    }
}

/// Pretty print the given vector of MSCCL experiment parameters as a table.
///
/// # Arguments
/// * `configs` - A vector of MSCCL experiment parameters to pretty print
pub fn pretty_print_configs(configs: &Vec<MscclExperimentParams>, color: bool) {
    let num_rows = configs.len();

    println!("|----------------------------+------------------------------------------------------------------------------------------------|");
    // println!("|----------------------------|----------|---------------|---------------------------------------|--------------|--------------|");
    println!("|         collective         |    op    |     dtype     |              algorithm:               |   channels   |    chunks    |");
    println!("|----------------------------+----------+---------------+---------------------------------------+--------------+--------------|");
    for (i, config) in configs.iter().enumerate() {

        if color {
            println!(
                "| collective: {}{:<14}{} | op: {}{:^4}{} | dtype: {}{:^6}{} | algorithm: {}{:^26}{} | channels: {}{:>2}{} | chunks: {}{:>4}{} |",
                color::Fg(color::Yellow),
                config.nc_collective,
                color::Fg(color::Reset),

                color::Fg(color::LightBlue),
                config.nc_op,
                color::Fg(color::Reset),

                color::Fg(color::LightYellow),
                config.nc_dtype,
                color::Fg(color::Reset),

                color::Fg(color::Magenta),
                config.algorithm,
                color::Fg(color::Reset),

                color::Fg(color::LightGreen),
                config.ms_channels,
                color::Fg(color::Reset),

                color::Fg(color::LightCyan),
                config.ms_chunks,
                color::Fg(color::Reset),
            );
        } else {
            println!(
                "| collective: {:<14} | op: {:^4} | dtype: {:^6} | algorithm: {:^26} | channels: {:>2} | chunks: {:>4} |",
                config.nc_collective,
                config.nc_op,
                config.nc_dtype,
                config.algorithm,
                config.ms_channels,
                config.ms_chunks,
            );
        }

        // Print the bottom line without dividing "plus" signs
        if i == num_rows - 1 {
            println!("|-----------------------------------------------------------------------------------------------------------------------------|");
        } else {
            println!("|----------------------------+----------+---------------+---------------------------------------+--------------+--------------|");
        }
    }
}

/// Pretty print the given vector of MSCCL experiment results as a table
/// 
/// # Arguments
/// * `entries` - A vector of MSCCL experiment results to pretty print
pub fn pretty_print_result_manifest(entries: &Vec<ManifestEntry>) {
    let mut table = prettytable::Table::new();

    // Add a title row
    table.add_row(row!["Collective", "Op", "DType", "Algorithm", "Num Channels", "Num Chunks", "Num GPUs", "Buffer Size Factor", "Overall Result"]);

    // Iterate over entries and add each as a row
    for entry in entries {
        let result_pretty = match entry.overall_result {
            ResultDescription::Success => format!("✅ {}", entry.overall_result),
            ResultDescription::PartialFailure => format!("⛓️‍💥 {}", entry.overall_result),
            ResultDescription::Failure => format!("❌ {}", entry.overall_result),
            ResultDescription::Skipped => format!("⏭️ {}", entry.overall_result),
            ResultDescription::Blacklisted => format!("💔 {}", entry.overall_result),
        };

        table.add_row(prettytable::Row::new(vec![
            prettytable::Cell::new(&entry.collective),
            prettytable::Cell::new(&entry.op),
            prettytable::Cell::new(&entry.dtype),
            prettytable::Cell::new(&entry.algorithm),
            prettytable::Cell::new(&entry.num_channels.to_string()),
            prettytable::Cell::new(&entry.num_chunks.to_string()),
            prettytable::Cell::new(&entry.num_gpus.to_string()),
            prettytable::Cell::new(&entry.buffer_size_factor.to_string()),
            prettytable::Cell::new(result_pretty.as_str()),
        ]));
    }

    // Print the table
    table.printstd();
}

/// Give the (probable) name of the XML file for a given set of experiment parameters
pub fn params_to_xml(
    collective: &str,
    comm_algorithm: &str,
    num_nodes: u64,
    num_gpus: u64,
    msccl_channels: u64,
    msccl_chunks: u64,
    gpu_as_node: bool,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // [HACK] Convert collective to Liuyao Format
    let converted_collective = match collective {
        "all-reduce" => "allreduce",
        "all-gather" => "allgather",
        "all-to-all" => "alltoall",
        "broadcast" => "broadcast",
        "gather" => "gather",
        "hypercube" => "hypercube",
        "reduce" => "reduce",
        "reduce-scatter" => "reducescatter",
        "scatter" => "scatter",
        "sendrecv" => "sendrecv",
        _ => {
            return Err(format!("Could not find a matching Ly-formatted collective for: {}", collective).into());
        }
    };

    // [HACK] Convert algorithm to Liuyao Format
    let converted_algo = match comm_algorithm {
        "binary-tree" => "binary_tree",
        "binomial-tree" => "binomial_tree",
        "recursive-doubling" => "recursive_doubling",
        "recursive-halving-doubling" => "recursive_doubling_halving",
        "ring" => "ring",
        "trinomial-tree" => "trinomial_tree",
        _ => {
            return Err(format!("Could not find a matching Ly-formatted comm. algorithm for: {}", comm_algorithm).into());
        }
    };

    // "Build" the XML file name
    // Note: We leave off op, datatype, and iteration because they do not apply to the algo XML
    Ok(PathBuf::from(format!(
        "{}_{}_node{}_gpu{}_mcl{}_mck{}_gan{}.xml",
        converted_collective,
        comm_algorithm,
        num_nodes,
        num_gpus,
        msccl_channels,
        msccl_chunks,
        if gpu_as_node { 1 } else { 0 }
    )))
}

pub fn verify_env() -> Result<(), Box<dyn std::error::Error>> {
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