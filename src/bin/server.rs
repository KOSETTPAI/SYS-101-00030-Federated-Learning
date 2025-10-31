#![allow(unused)]

// The full gRPC server requires external crates (candle_*, tonic service impls).
// To keep default builds working smoothly, enable it with `--features grpc`.
#[cfg(feature = "grpc")]
use federated_learning::common::*;
#[cfg(feature = "grpc")]
use federated_learning::data::*;
#[cfg(feature = "grpc")]
use std::collections::HashMap;
#[cfg(feature = "grpc")]
use std::net::SocketAddr;
#[cfg(feature = "grpc")]
use std::sync::{Arc, RwLock};
#[cfg(feature = "grpc")]
use tonic::{transport::Server, Request, Response, Status};
#[cfg(feature = "grpc")]
use clap::Parser;
#[cfg(feature = "grpc")]
use log::{info, warn, error};
#[cfg(feature = "grpc")]
use ndarray::{Array1, Array2};

#[cfg(feature = "grpc")]
#[derive(Parser)]
#[command(name = "federated-server")]
#[command(about = "A federated learning parameter server")]
struct Args {
    #[arg(long, default_value = "127.0.0.1:50051")]
    address: String,
    
    #[arg(long, default_value = "1.0")]
    learning_rate: f64,
    
    #[arg(long, default_value = "10")]
    epochs_per_round: usize,
}

#[cfg(feature = "grpc")]
#[derive(Debug)]
struct ParameterServerState {
    models: RwLock<HashMap<String, ModelState>>,
    // Dataset loading via candle_* is disabled in default builds
    training_config: TrainingArgs,
}

#[cfg(feature = "grpc")]
impl ParameterServerState {
    fn new(training_config: TrainingArgs) -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            training_config,
        }
    }
}

#[cfg(feature = "grpc")]
struct ParameterServerImpl {
    state: Arc<ParameterServerState>,
}

#[cfg(feature = "grpc")]
impl ParameterServerImpl {
    fn new(state: Arc<ParameterServerState>) -> Self {
        Self { state }
    }

    fn create_initial_model(&self, model_name: &str) -> Result<ModelParameters, Status> {
        let model = LinearModel::new().map_err(|e| {
            Status::internal(format!("Failed to create model: {}", e))
        })?;

        model.get_parameters().map_err(|e| {
            Status::internal(format!("Failed to get model parameters: {}", e))
        })
    }

    async fn perform_federated_round(&self, model_name: &str) -> Result<(), Status> {
        let clients = {
            let models = self.state.models.read().unwrap();
            let model_state = models.get(model_name).ok_or_else(|| {
                Status::not_found(format!("Model '{}' not found", model_name))
            })?;
            
            if model_state.status != ModelStatus::Ready {
                return Err(Status::failed_precondition("Model is not ready for training"));
            }
            
            model_state.clients.clone()
        };

        if clients.is_empty() {
            return Err(Status::failed_precondition("No clients registered"));
        }

        info!("Starting federated training round for model '{}'", model_name);
        
        // Update status to training
        {
            let mut models = self.state.models.write().unwrap();
            if let Some(model_state) = models.get_mut(model_name) {
                model_state.status = ModelStatus::Training;
            }
        }

        // Get current global model parameters
        let global_params = {
            let models = self.state.models.read().unwrap();
            models.get(model_name).unwrap().parameters.clone().unwrap()
        };

        // Simulate client training (in a real implementation, you would call clients via gRPC)
        let client_params = self.simulate_client_training(model_name, &global_params, &clients).await?;
        
        // Perform federated averaging
        let averaged_params = federate_average(client_params).map_err(|e| {
            Status::internal(format!("Failed to average parameters: {}", e))
        })?;

        // Update global model
        {
            let mut models = self.state.models.write().unwrap();
            if let Some(model_state) = models.get_mut(model_name) {
                model_state.parameters = Some(averaged_params);
                model_state.status = ModelStatus::Ready;
                model_state.current_round += 1;
            }
        }

        info!("Completed federated training round for model '{}'", model_name);
        Ok(())
    }

    async fn simulate_client_training(
        &self,
        model_name: &str,
        global_params: &ModelParameters,
        clients: &[ClientInfo],
    ) -> Result<Vec<ModelParameters>, Status> {
        let dataset = MnistData::load().await.map_err(|e| {
            Status::internal(format!("Failed to load MNIST dataset: {}", e))
        })?;

        let train_images = dataset.train_images;
        let train_labels = dataset.train_labels;
        let test_images = dataset.test_images;
        let test_labels = dataset.test_labels;

        let total_samples = train_images.shape().dims()[0];
        let samples_per_client = total_samples / clients.len();
        
        let mut client_params = Vec::new();
        
        for (i, client) in clients.iter().enumerate() {
            info!("Training model for client: {}", client.id);
            
            // Create model with global parameters
            let mut model = LinearModel::new().map_err(|e| {
                Status::internal(format!("Failed to create client model: {}", e))
            })?;
            
            // Set global parameters
            model.set_parameters(global_params).map_err(|e| {
                Status::internal(format!("Failed to set model parameters: {}", e))
            })?;

            // Get client's data slice (IID distribution)
            let start_idx = i * samples_per_client;
            let end_idx = if i == clients.len() - 1 {
                total_samples // Last client gets remaining samples
            } else {
                (i + 1) * samples_per_client
            };

            let client_train_images = train_images.i(start_idx..end_idx).unwrap();
            let client_train_labels = train_labels.i(start_idx..end_idx).unwrap();
            let client_test_images = test_images.i(start_idx..end_idx).unwrap();
            let client_test_labels = test_labels.i(start_idx..end_idx).unwrap();

            // Train the model locally using ndarray-based trainer
            model.train(
                &client_train_images,
                &client_train_labels,
                self.state.training_config.learning_rate as f32,
                self.state.training_config.epochs,
            ).map_err(|e| Status::internal(format!("Failed to train client model: {}", e)))?;

            // Get updated parameters
            let updated_params = model.get_parameters().map_err(|e| {
                Status::internal(format!("Failed to get updated parameters: {}", e))
            })?;

            client_params.push(updated_params);
        }

        Ok(client_params)
    }
}

#[cfg(feature = "grpc")]
#[tonic::async_trait]
impl ParameterServer for ParameterServerImpl {
    async fn register(
        &self,
        request: Request<RegisterRequest>,
    ) -> Result<Response<RegisterResponse>, Status> {
        let req = request.into_inner();
        info!("Client registration request: {} for model {}", req.client_id, req.model_name);

        let mut models = self.state.models.write().unwrap();
        let model_state = models.entry(req.model_name.clone()).or_insert_with(ModelState::new);

        // Check if client already registered
        if model_state.clients.iter().any(|c| c.id == req.client_id) {
            return Ok(Response::new(RegisterResponse {
                success: false,
                message: "Client already registered".to_string(),
            }));
        }

        // Add client
        model_state.clients.push(ClientInfo {
            id: req.client_id.clone(),
            address: req.client_ip,
            status: ModelStatus::Initialized,
        });

        info!("Registered client {} for model {}", req.client_id, req.model_name);

        Ok(Response::new(RegisterResponse {
            success: true,
            message: format!("Client {} registered successfully", req.client_id),
        }))
    }

    async fn init(
        &self,
        request: Request<InitRequest>,
    ) -> Result<Response<InitResponse>, Status> {
        let req = request.into_inner();
        info!("Model initialization request for: {}", req.model_name);

        let initial_params = self.create_initial_model(&req.model_name)?;

        let mut models = self.state.models.write().unwrap();
        let model_state = models.entry(req.model_name.clone()).or_insert_with(ModelState::new);
        model_state.parameters = Some(initial_params);
        model_state.status = ModelStatus::Ready;
        model_state.current_round = 0;

        info!("Initialized model: {}", req.model_name);

        Ok(Response::new(InitResponse {
            success: true,
            message: format!("Model {} initialized successfully", req.model_name),
        }))
    }

    async fn train(
        &self,
        request: Request<TrainRequest>,
    ) -> Result<Response<TrainResponse>, Status> {
        let req = request.into_inner();
        info!("Training request for model: {} with {} rounds", req.model_name, req.rounds);

        let mut completed_rounds = 0;
        
        for round in 0..req.rounds {
            match self.perform_federated_round(&req.model_name).await {
                Ok(()) => {
                    completed_rounds = round + 1;
                    info!("Completed training round {}/{} for model {}", 
                          completed_rounds, req.rounds, req.model_name);
                }
                Err(e) => {
                    error!("Failed training round {}: {}", round + 1, e);
                    return Ok(Response::new(TrainResponse {
                        success: false,
                        message: format!("Training failed at round {}: {}", round + 1, e),
                        completed_rounds,
                    }));
                }
            }
        }

        Ok(Response::new(TrainResponse {
            success: true,
            message: format!("Completed {} training rounds", completed_rounds),
            completed_rounds,
        }))
    }

    async fn get_model(
        &self,
        request: Request<GetModelRequest>,
    ) -> Result<Response<GetModelResponse>, Status> {
        let req = request.into_inner();
        
        let models = self.state.models.read().unwrap();
        let model_state = models.get(&req.model_name).ok_or_else(|| {
            Status::not_found(format!("Model '{}' not found", req.model_name))
        })?;

        Ok(Response::new(GetModelResponse {
            success: true,
            parameters: model_state.parameters.clone(),
            status: model_state.status.clone().into(),
            message: format!("Model {} retrieved", req.model_name),
        }))
    }

    async fn test_model(
        &self,
        request: Request<TestModelRequest>,
    ) -> Result<Response<TestModelResponse>, Status> {
        let req = request.into_inner();
        info!("Test request for model: {}", req.model_name);

        let model_params = {
            let models = self.state.models.read().unwrap();
            let model_state = models.get(&req.model_name).ok_or_else(|| {
                Status::not_found(format!("Model '{}' not found", req.model_name))
            })?;

            model_state.parameters.clone().ok_or_else(|| {
                Status::failed_precondition("Model not initialized")
            })?
        };

        // Load test dataset and compute accuracy
        let dataset = MnistData::load().await.map_err(|e| {
            Status::internal(format!("Failed to load dataset: {}", e))
        })?;

        let test_images = dataset.test_images;
        let test_labels = dataset.test_labels;

        let mut model = LinearModel::new().map_err(|e| {
            Status::internal(format!("Failed to create model: {}", e))
        })?;

        model.set_parameters(&model_params).map_err(|e| {
            Status::internal(format!("Failed to set parameters: {}", e))
        })?;

        let accuracy = calculate_accuracy(&model, &test_images, &test_labels).map_err(|e| {
            Status::internal(format!("Failed to calculate accuracy: {}", e))
        })?;

        info!("Model {} test accuracy: {:.2}%", req.model_name, accuracy * 100.0);

        Ok(Response::new(TestModelResponse {
            success: true,
            accuracy,
            message: format!("Test accuracy: {:.2}%", accuracy * 100.0),
        }))
    }
}

#[cfg(feature = "grpc")]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    let args = Args::parse();
    let addr: SocketAddr = args.address.parse()?;
    
    let training_config = TrainingArgs {
        learning_rate: args.learning_rate,
        epochs: args.epochs_per_round,
    };
    
    let state = Arc::new(ParameterServerState::new(training_config));
    let server_impl = ParameterServerImpl::new(state);

    info!("Starting Parameter Server on {}", addr);

    // Note: ParameterServerServer here is a placeholder; full tonic service glue is
    // behind the `grpc` feature in downstream code.
    Server::builder()
        .add_service(ParameterServerServer::new(server_impl))
        .serve(addr)
        .await?;

    Ok(())
}

#[cfg(not(feature = "grpc"))]
fn main() {
    println!("server binary built without 'grpc' feature. Enable with: cargo run --bin server --features grpc");
}