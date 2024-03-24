use regex::Regex;
use polars::prelude::*;

// mod util;
use crate::{Row, Permutation, MscclExperimentParams};

/// Convert rows to a Polars DataFrame
/// 
/// Note: The implementaiton is very manual and not efficient.
pub fn rows_to_df(rows: Vec<Row>) -> Result<DataFrame, Box<dyn std::error::Error>> {
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
pub fn parse_line(line: &str) -> Result<Option<Row>, Box<dyn std::error::Error>> {
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