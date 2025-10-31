#![allow(unused)]
/// Federated Learning Demo
/// 
/// This example demonstrates how to use the federated learning system:
/// 1. Start a parameter server
/// 2. Initialize a global model
/// 3. Register clients
/// 4. Run federated training rounds
/// 5. Test the final model

use anyhow::Result;
use log::info;
use std::process::Command;
use std::time::Duration;
use tokio::time::sleep;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::init();
    
    println!("🚀 Federated Learning Demo");
    println!("========================");
    println!();
    
    println!("This demo will show you how to:");
    println!("1. Start a parameter server on port 50051");
    println!("2. Start multiple clients that connect to the server");
    println!("3. Initialize a global MNIST model");
    println!("4. Register clients for federated learning");
    println!("5. Run federated training rounds");
    println!("6. Test the final model accuracy");
    println!();
    
    println!("To run this demo manually, open multiple terminals:");
    println!();
    
    println!("Terminal 1 (Parameter Server):");
    println!("$ cargo run --bin server --release -- --port 50051");
    println!();
    
    println!("Terminal 2 (Client 1):");
    println!("$ cargo run --bin client --release -- --server-address 127.0.0.1:50051 --model-name mnist");
    println!("client> join");
    println!("client> server-init");
    println!();
    
    println!("Terminal 3 (Client 2):");
    println!("$ cargo run --bin client --release -- --server-address 127.0.0.1:50051 --model-name mnist");
    println!("client> join");
    println!();
    
    println!("Terminal 4 (Client 3):");
    println!("$ cargo run --bin client --release -- --server-address 127.0.0.1:50051 --model-name mnist");
    println!("client> join");
    println!();
    
    println!("Back to Terminal 2 (Start Training):");
    println!("client> server-train 5");
    println!("client> server-test");
    println!();
    
    println!("📚 Commands available in client interactive mode:");
    println!("- join: Register with the server");
    println!("- train: Train local model");
    println!("- test: Test local model");
    println!("- get: Get global model from server");
    println!("- server-init: Initialize global model on server");
    println!("- server-train <rounds>: Start federated training");
    println!("- server-test: Test global model on server");
    println!("- quit: Exit client");
    
    Ok(())
}

/// Helper function to check if a command exists
fn command_exists(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

/// Helper function to start server in background (for automated testing)
#[allow(dead_code)]
async fn start_server_background() -> Result<std::process::Child> {
    let child = Command::new("cargo")
        .args(&["run", "--bin", "server", "--release", "--", "--port", "50051"])
        .spawn()?;
    
    // Give the server time to start
    sleep(Duration::from_secs(3)).await;
    
    Ok(child)
}

/// Helper function to run a client command
#[allow(dead_code)]
async fn run_client_command(command: &str) -> Result<String> {
    let output = Command::new("cargo")
        .args(&["run", "--bin", "client", "--release"])
        .arg(command)
        .output()?;
    
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}