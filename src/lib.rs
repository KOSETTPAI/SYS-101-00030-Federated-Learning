#![allow(unused)]
pub mod common;
pub mod data;

// Generated gRPC/protobuf types
pub mod federated_learning {
    tonic::include_proto!("federated_learning");
}

pub use federated_learning::*;
