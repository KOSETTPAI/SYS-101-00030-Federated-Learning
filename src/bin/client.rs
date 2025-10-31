#![allow(unused)]
use federated_learning::common::*;
use federated_learning::data::*;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tonic::{transport::Channel, Request, Response, Status};
use clap::Parser;
use log::{info, warn, error};
use uuid::Uuid;
use ndarray::{Array1, Array2};

#[derive(Parser)]
#[command(name = "federated-client")]
#[command(about = "A federated learning client")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:50051")]
    server_address: String,
    
    #[arg(long, default_value = "127.0.0.1:50052")]
    client_address: String,
    
    #[arg(long, default_value = "mnist")]
    model_name: String,
    
    #[arg(long, default_value = "1.0")]
    learning_rate: f64,
    
    #[arg(long, default_value = "5")]
    local_epochs: usize,
    
    #[arg(long)]
    client_id: Option<String>,
}

#[derive(Debug)]
struct ClientState {
    models: RwLock<HashMap<String, LinearModel>>,
    training_config: TrainingArgs,
    client_id: String,
    server_address: String,
}

impl ClientState {
    fn new(training_config: TrainingArgs, client_id: String, server_address: String) -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            training_config,
            client_id,
            server_address,
        }
    }
}

struct FederatedClientImpl {
    state: Arc<ClientState>,
}

impl FederatedClientImpl {
    fn new(state: Arc<ClientState>) -> Self {
        Self { state }
    }

    async fn get_server_client(&self) -> Result<ParameterServerClient<Channel>, Box<dyn std::error::Error>> {
        let channel = tonic::transport::Channel::from_shared(format!("http://{}", self.state.server_address))?
            .connect()
            .await?;
        Ok(ParameterServerClient::new(channel))
    }

    async fn join_server(&self, model_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut client = self.get_server_client().await?;
        
        let request = Request::new(RegisterRequest {
            client_id: self.state.client_id.clone(),
            client_ip: "127.0.0.1:50052".to_string(), // TODO: Use actual client address
            model_name: model_name.to_string(),
        });

        let response = client.register(request).await?;
        let response = response.into_inner();

        if !response.success {
            return Err(format!("Failed to register: {}", response.message).into());
        }

        info!("Successfully joined server for model: {}", model_name);
        Ok(())
    }

    async fn get_global_model(&self, model_name: &str) -> Result<ModelParameters, Box<dyn std::error::Error>> {
        let mut client = self.get_server_client().await?;
        
        let request = Request::new(GetModelRequest {
            model_name: model_name.to_string(),
        });

        let response = client.get_model(request).await?;
        let response = response.into_inner();

        if !response.success {
            return Err(format!("Failed to get model: {}", response.message).into());
        }

        response.parameters.ok_or("No model parameters received".into())
    }

    async fn train_local_model(&self, model_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        info!("Starting local training for model: {}", model_name);

        // Get global model parameters from server
        let global_params = self.get_global_model(model_name).await?;

        // Load MNIST dataset
        let dataset = candle_datasets::vision::mnist::load()?;
        let dev = Device::cuda_if_available(0)?;

        let train_labels = dataset.train_labels.to_dtype(DType::U32)?.to_device(&dev)?;
        let train_images = dataset.train_images.to_device(&dev)?;
        let test_images = dataset.test_images.to_device(&dev)?;
        let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

        // Create local model and set global parameters
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
        let mut model = LinearModel::new(vs)?;
        model.set_parameters(&global_params)?;

        // For simplicity, train on full dataset (in practice, each client would have subset)
        let sgd = candle_nn::SGD::new(varmap.all_vars(), self.state.training_config.learning_rate)?;
        let trained_model = train_model(
            model,
            &test_images,
            &test_labels,
            &train_images,
            &train_labels,
            sgd,
            self.state.training_config.epochs,
        )?;

        // Store the trained model
        {
            let mut models = self.state.models.write().unwrap();
            models.insert(model_name.to_string(), trained_model);
        }

        info!("Completed local training for model: {}", model_name);
        Ok(())
    }

    async fn test_local_model(&self, model_name: &str) -> Result<f32, Box<dyn std::error::Error>> {
        let model = {
            let models = self.state.models.read().unwrap();
            models.get(model_name).cloned().ok_or("Model not found")?
        };

        // Load test dataset
        let dataset = candle_datasets::vision::mnist::load()?;
        let dev = Device::cuda_if_available(0)?;

        let test_images = dataset.test_images.to_device(&dev)?;
        let test_labels = dataset.test_labels.to_dtype(DType::U32)?.to_device(&dev)?;

        let accuracy = calculate_accuracy(&model, &test_images, &test_labels)?;
        info!("Local model {} test accuracy: {:.2}%", model_name, accuracy * 100.0);
        
        Ok(accuracy)
    }
}

#[tonic::async_trait]
impl FederatedClient for FederatedClientImpl {
    async fn train_local(
        &self,
        request: Request<ClientTrainRequest>,
    ) -> Result<Response<ClientTrainResponse>, Status> {
        let req = request.into_inner();
        info!("Received local training request for model: {}", req.model_name);

        // This would be called by the server in a fully distributed implementation
        // For now, we'll return a simple response
        match self.train_local_model(&req.model_name).await {
            Ok(()) => {
                let model = {
                    let models = self.state.models.read().unwrap();
                    models.get(&req.model_name).cloned()
                };

                if let Some(model) = model {
                    match model.get_parameters() {
                        Ok(params) => {
                            Ok(Response::new(ClientTrainResponse {
                                success: true,
                                updated_parameters: Some(params),
                                message: "Local training completed".to_string(),
                            }))
                        }
                        Err(e) => {
                            error!("Failed to get model parameters: {}", e);
                            Ok(Response::new(ClientTrainResponse {
                                success: false,
                                updated_parameters: None,
                                message: format!("Failed to get parameters: {}", e),
                            }))
                        }
                    }
                } else {
                    Ok(Response::new(ClientTrainResponse {
                        success: false,
                        updated_parameters: None,
                        message: "Model not found after training".to_string(),
                    }))
                }
            }
            Err(e) => {
                error!("Local training failed: {}", e);
                Ok(Response::new(ClientTrainResponse {
                    success: false,
                    updated_parameters: None,
                    message: format!("Training failed: {}", e),
                }))
            }
        }
    }

    async fn get_local_model(
        &self,
        request: Request<GetModelRequest>,
    ) -> Result<Response<GetModelResponse>, Status> {
        let req = request.into_inner();
        
        let models = self.state.models.read().unwrap();
        let model = models.get(&req.model_name);

        match model {
            Some(model) => {
                match model.get_parameters() {
                    Ok(params) => {
                        Ok(Response::new(GetModelResponse {
                            success: true,
                            parameters: Some(params),
                            status: TrainingStatus::Ready,
                            message: "Model retrieved".to_string(),
                        }))
                    }
                    Err(e) => {
                        Err(Status::internal(format!("Failed to get parameters: {}", e)))
                    }
                }
            }
            None => {
                Err(Status::not_found("Model not found"))
            }
        }
    }

    async fn test_local(
        &self,
        request: Request<TestModelRequest>,
    ) -> Result<Response<TestModelResponse>, Status> {
        let req = request.into_inner();
        
        match self.test_local_model(&req.model_name).await {
            Ok(accuracy) => {
                Ok(Response::new(TestModelResponse {
                    success: true,
                    accuracy,
                    message: format!("Test accuracy: {:.2}%", accuracy * 100.0),
                }))
            }
            Err(e) => {
                error!("Failed to test model: {}", e);
                Ok(Response::new(TestModelResponse {
                    success: false,
                    accuracy: 0.0,
                    message: format!("Test failed: {}", e),
                }))
            }
        }
    }
}

async fn run_interactive_client(client_impl: Arc<FederatedClientImpl>, model_name: String) -> Result<(), Box<dyn std::error::Error>> {
    use std::io::{self, Write};

    println!("Federated Learning Client Interactive Mode");
    println!("Available commands:");
    println!("  join - Join the federated learning network");
    println!("  train - Train the local model");
    println!("  test - Test the local model");
    println!("  get - Get model information from server");
    println!("  server-test - Test the global model on server");
    println!("  server-init - Initialize model on server");
    println!("  server-train <rounds> - Start federated training on server");
    println!("  quit - Exit the client");
    println!();

    loop {
        print!("client> ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        let parts: Vec<&str> = input.split_whitespace().collect();
        let command = parts[0];

        match command {
            "join" => {
                match client_impl.join_server(&model_name).await {
                    Ok(()) => println!("✓ Successfully joined the network"),
                    Err(e) => println!("✗ Failed to join: {}", e),
                }
            }
            "train" => {
                match client_impl.train_local_model(&model_name).await {
                    Ok(()) => println!("✓ Local training completed"),
                    Err(e) => println!("✗ Training failed: {}", e),
                }
            }
            "test" => {
                match client_impl.test_local_model(&model_name).await {
                    Ok(accuracy) => println!("✓ Local model accuracy: {:.2}%", accuracy * 100.0),
                    Err(e) => println!("✗ Test failed: {}", e),
                }
            }
            "get" => {
                match client_impl.get_global_model(&model_name).await {
                    Ok(_) => println!("✓ Retrieved global model"),
                    Err(e) => println!("✗ Failed to get model: {}", e),
                }
            }
            "server-test" => {
                match client_impl.get_server_client().await {
                    Ok(mut server_client) => {
                        let request = Request::new(TestModelRequest {
                            model_name: model_name.clone(),
                        });
                        match server_client.test_model(request).await {
                            Ok(response) => {
                                let resp = response.into_inner();
                                if resp.success {
                                    println!("✓ Server model accuracy: {:.2}%", resp.accuracy * 100.0);
                                } else {
                                    println!("✗ Server test failed: {}", resp.message);
                                }
                            }
                            Err(e) => println!("✗ Server test failed: {}", e),
                        }
                    }
                    Err(e) => println!("✗ Failed to connect to server: {}", e),
                }
            }
            "server-init" => {
                match client_impl.get_server_client().await {
                    Ok(mut server_client) => {
                        let request = Request::new(InitRequest {
                            model_name: model_name.clone(),
                        });
                        match server_client.init(request).await {
                            Ok(response) => {
                                let resp = response.into_inner();
                                if resp.success {
                                    println!("✓ Server model initialized");
                                } else {
                                    println!("✗ Server init failed: {}", resp.message);
                                }
                            }
                            Err(e) => println!("✗ Server init failed: {}", e),
                        }
                    }
                    Err(e) => println!("✗ Failed to connect to server: {}", e),
                }
            }
            "server-train" => {
                let rounds = if parts.len() > 1 {
                    parts[1].parse::<i32>().unwrap_or(1)
                } else {
                    1
                };
                
                match client_impl.get_server_client().await {
                    Ok(mut server_client) => {
                        let request = Request::new(TrainRequest {
                            model_name: model_name.clone(),
                            rounds,
                        });
                        println!("Starting {} federated training rounds...", rounds);
                        match server_client.train(request).await {
                            Ok(response) => {
                                let resp = response.into_inner();
                                if resp.success {
                                    println!("✓ Federated training completed: {} rounds", resp.completed_rounds);
                                } else {
                                    println!("✗ Federated training failed: {}", resp.message);
                                }
                            }
                            Err(e) => println!("✗ Federated training failed: {}", e),
                        }
                    }
                    Err(e) => println!("✗ Failed to connect to server: {}", e),
                }
            }
            "quit" => {
                println!("Goodbye!");
                break;
            }
            _ => {
                println!("Unknown command: {}", command);
            }
        }
        println!();
    }

    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let args = Args::parse();
    
    let client_id = args.client_id.unwrap_or_else(|| {
        format!("client-{}", Uuid::new_v4().to_string()[..8].to_string())
    });
    
    let training_config = TrainingArgs {
        learning_rate: args.learning_rate,
        epochs: args.local_epochs,
    };
    
    let state = Arc::new(ClientState::new(
        training_config,
        client_id.clone(),
        args.server_address.clone(),
    ));
    
    let client_impl = Arc::new(FederatedClientImpl::new(state));

    info!("Starting Federated Client: {}", client_id);
    info!("Server: {}", args.server_address);
    info!("Model: {}", args.model_name);

    // Run interactive client
    run_interactive_client(client_impl, args.model_name).await?;

    Ok(())
}