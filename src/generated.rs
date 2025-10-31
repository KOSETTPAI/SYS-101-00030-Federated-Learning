// Manual protobuf implementation to avoid protoc dependency
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tonic::{Request, Response, Status, transport::Server};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelParameters {
    pub weights: Vec<f32>,
    pub biases: Vec<f32>,
    pub weight_shape: Vec<i32>,
    pub bias_shape: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrainingStatus {
    Initialized = 0,
    Training = 1,
    Ready = 2,
}

impl From<i32> for TrainingStatus {
    fn from(value: i32) -> Self {
        match value {
            0 => TrainingStatus::Initialized,
            1 => TrainingStatus::Training,
            2 => TrainingStatus::Ready,
            _ => TrainingStatus::Initialized,
        }
    }
}

impl From<TrainingStatus> for i32 {
    fn from(status: TrainingStatus) -> Self {
        status as i32
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RegisterRequest {
    pub client_id: String,
    pub client_ip: String,
    pub model_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RegisterResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InitRequest {
    pub model_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InitResponse {
    pub success: bool,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainRequest {
    pub model_name: String,
    pub rounds: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainResponse {
    pub success: bool,
    pub message: String,
    pub completed_rounds: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetModelRequest {
    pub model_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct GetModelResponse {
    pub success: bool,
    pub parameters: Option<ModelParameters>,
    pub status: TrainingStatus,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestModelRequest {
    pub model_name: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestModelResponse {
    pub success: bool,
    pub accuracy: f32,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClientTrainRequest {
    pub model_name: String,
    pub global_parameters: ModelParameters,
    pub train_data: Vec<f32>,
    pub train_labels: Vec<i32>,
    pub epochs: i32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ClientTrainResponse {
    pub success: bool,
    pub updated_parameters: Option<ModelParameters>,
    pub message: String,
}

// Trait definitions for gRPC services
#[tonic::async_trait]
pub trait ParameterServer: Send + Sync + 'static {
    async fn register(&self, request: Request<RegisterRequest>) -> Result<Response<RegisterResponse>, Status>;
    async fn init(&self, request: Request<InitRequest>) -> Result<Response<InitResponse>, Status>;
    async fn train(&self, request: Request<TrainRequest>) -> Result<Response<TrainResponse>, Status>;
    async fn get_model(&self, request: Request<GetModelRequest>) -> Result<Response<GetModelResponse>, Status>;
    async fn test_model(&self, request: Request<TestModelRequest>) -> Result<Response<TestModelResponse>, Status>;
}

#[tonic::async_trait]
pub trait FederatedClient: Send + Sync + 'static {
    async fn train_local(&self, request: Request<ClientTrainRequest>) -> Result<Response<ClientTrainResponse>, Status>;
    async fn get_local_model(&self, request: Request<GetModelRequest>) -> Result<Response<GetModelResponse>, Status>;
    async fn test_local(&self, request: Request<TestModelRequest>) -> Result<Response<TestModelResponse>, Status>;
}

// Server implementations
pub struct ParameterServerServer<T> {
    inner: T,
}

impl<T> ParameterServerServer<T>
where
    T: ParameterServer,
{
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

pub struct FederatedClientServer<T> {
    inner: T,
}

impl<T> FederatedClientServer<T>
where
    T: FederatedClient,
{
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}

// Client implementations
pub struct ParameterServerClient<T> {
    inner: T,
}

impl<T> ParameterServerClient<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }
}