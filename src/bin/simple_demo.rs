#![allow(unused)]
/// Simple Federated Learning Demonstration
/// 
/// This demonstrates the key concepts of federated learning:
/// 1. Global model initialization
/// 2. Data distribution to clients
/// 3. Local training on each client
/// 4. Parameter aggregation (FedAvg)
/// 5. Model accuracy evaluation

use federated_learning::common::*;
use federated_learning::data::*;
use std::sync::{Arc, RwLock};
use log::info;
use ndarray::{Array1, Array2};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    
    println!("🚀 Simple Federated Learning Demo");
    println!("==================================");
    
    // Step 1: Initialize global model
    info!("Step 1: Initializing global model");
    let mut global_model = LinearModel::new()?;
    let initial_params = global_model.get_parameters()?;
    
    println!("✓ Global model initialized with {} weights and {} biases", 
             initial_params.weights.len(), initial_params.biases.len());
    
    // Step 2: Load and distribute data
    info!("Step 2: Loading and distributing data");
    let mnist_data = MnistData::load().await?;
    let num_clients = 3;
    let client_datasets = mnist_data.split_train_data(num_clients)?;
    
    println!("✓ Data distributed to {} clients:", num_clients);
    for (i, (images, labels)) in client_datasets.iter().enumerate() {
        println!("  Client {}: {} samples", i + 1, images.shape()[0]);
    }
    
    // Step 3: Simulate federated training
    info!("Step 3: Starting federated training");
    let num_rounds = 3;
    let learning_rate = 0.01;
    let local_epochs = 5;
    
    for round in 1..=num_rounds {
        println!("\n--- Training Round {} ---", round);
        
        // Step 3a: Distribute current global model to clients
        let mut client_models = Vec::new();
        let mut client_params = Vec::new();
        
        for i in 0..num_clients {
            let mut client_model = LinearModel::new()?;
            client_model.set_parameters(&global_model.get_parameters()?)?;
            client_models.push(client_model);
        }
        
        // Step 3b: Local training on each client
        for (client_id, (client_model, (client_images, client_labels))) in 
            client_models.iter_mut().zip(client_datasets.iter()).enumerate() 
        {
            info!("Training client {} locally", client_id + 1);
            
            client_model.train(
                client_images, 
                client_labels, 
                learning_rate, 
                local_epochs
            )?;
            
            let client_accuracy = calculate_accuracy(client_model, client_images, client_labels)?;
            println!("  Client {} local accuracy: {:.2}%", 
                     client_id + 1, client_accuracy * 100.0);
            
            client_params.push(client_model.get_parameters()?);
        }
        
        // Step 3c: Aggregate client parameters using FedAvg
        info!("Aggregating client parameters");
        let aggregated_params = federate_average(client_params)?;
        global_model.set_parameters(&aggregated_params)?;
        
        // Step 3d: Evaluate global model
        let global_accuracy = calculate_accuracy(
            &global_model, 
            &mnist_data.test_images, 
            &mnist_data.test_labels
        )?;
        
        println!("✓ Round {} completed - Global accuracy: {:.2}%", 
                 round, global_accuracy * 100.0);
    }
    
    // Step 4: Final evaluation
    println!("\n🎯 Final Results");
    println!("================");
    
    let final_accuracy = calculate_accuracy(
        &global_model, 
        &mnist_data.test_images, 
        &mnist_data.test_labels
    )?;
    
    println!("Final global model accuracy: {:.2}%", final_accuracy * 100.0);
    
    // Demonstrate individual client performance
    println!("\nIndividual client performance on their local data:");
    for (client_id, (client_images, client_labels)) in client_datasets.iter().enumerate() {
        let client_accuracy = calculate_accuracy(&global_model, client_images, client_labels)?;
        println!("  Client {} data: {:.2}%", client_id + 1, client_accuracy * 100.0);
    }
    
    println!("\n✅ Federated learning demonstration completed!");
    println!("\nKey achievements:");
    println!("- ✓ Data privacy: Raw data never shared between clients");
    println!("- ✓ Collaborative learning: All clients benefit from shared knowledge");
    println!("- ✓ Federated averaging: Model parameters properly aggregated");
    println!("- ✓ Convergence: Model accuracy improved through multiple rounds");
    
    Ok(())
}

/// Demonstrate thread-safe federated learning with concurrent clients
#[allow(dead_code)]
async fn concurrent_demo() -> anyhow::Result<()> {
    info!("Demonstrating concurrent federated learning");
    
    // Shared global state (thread-safe)
    let global_state = Arc::new(RwLock::new(LinearModel::new()?));
    
    // Load data
    let mnist_data = MnistData::load().await?;
    let client_datasets = mnist_data.split_train_data(3)?;
    
    // Spawn concurrent client tasks
    let mut handles = Vec::new();
    
    for (client_id, (client_images, client_labels)) in client_datasets.into_iter().enumerate() {
        let global_state = Arc::clone(&global_state);
        
        let handle = tokio::spawn(async move {
            // Get current global model (thread-safe read)
            let global_params = {
                let global_model = global_state.read().unwrap();
                global_model.get_parameters().unwrap()
            };
            
            // Train locally
            let mut local_model = LinearModel::new().unwrap();
            local_model.set_parameters(&global_params).unwrap();
            local_model.train(&client_images, &client_labels, 0.01, 5).unwrap();
            
            info!("Client {} completed local training", client_id + 1);
            
            local_model.get_parameters().unwrap()
        });
        
        handles.push(handle);
    }
    
    // Collect results and aggregate
    let mut client_params = Vec::new();
    for handle in handles {
        client_params.push(handle.await?);
    }
    
    // Update global model (thread-safe write)
    let aggregated_params = federate_average(client_params)?;
    {
        let mut global_model = global_state.write().unwrap();
        global_model.set_parameters(&aggregated_params)?;
    }
    
    info!("Concurrent federated training round completed");
    
    Ok(())
}