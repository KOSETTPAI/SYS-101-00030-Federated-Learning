use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use ndarray::{Array1, Array2};
use rand::Rng;

// Re-export generated proto types
pub use crate::federated_learning::*;

pub const IMAGE_DIM: usize = 784;
pub const LABELS: usize = 10;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ModelStatus {
    Initialized,
    Training,
    Ready,
}

impl From<ModelStatus> for TrainingStatus {
    fn from(status: ModelStatus) -> Self {
        match status {
            ModelStatus::Initialized => TrainingStatus::Initialized,
            ModelStatus::Training => TrainingStatus::Training,
            ModelStatus::Ready => TrainingStatus::Ready,
        }
    }
}

impl From<TrainingStatus> for ModelStatus {
    fn from(status: TrainingStatus) -> Self {
        match status {
            TrainingStatus::Initialized => ModelStatus::Initialized,
            TrainingStatus::Training => ModelStatus::Training,
            TrainingStatus::Ready => ModelStatus::Ready,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClientInfo {
    pub id: String,
    pub address: String,
    pub status: ModelStatus,
}

#[derive(Debug, Clone)]
pub struct ModelState {
    pub parameters: Option<ModelParameters>,
    pub status: ModelStatus,
    pub clients: Vec<ClientInfo>,
    pub current_round: i32,
}

impl ModelState {
    pub fn new() -> Self {
        Self {
            parameters: None,
            status: ModelStatus::Initialized,
            clients: Vec::new(),
            current_round: 0,
        }
    }
}

pub trait Model: Sized + Send + Sync + Clone {
    fn new() -> anyhow::Result<Self>;
    fn forward(&self, xs: &Array2<f32>) -> anyhow::Result<Array2<f32>>;
    fn set_parameters(&mut self, params: &ModelParameters) -> anyhow::Result<()>;
    fn get_parameters(&self) -> anyhow::Result<ModelParameters>;
    fn train(&mut self, x: &Array2<f32>, y: &Array1<i32>, learning_rate: f32, epochs: usize) -> anyhow::Result<()>;
}

#[derive(Debug, Clone)]
pub struct LinearModel {
    weights: Array2<f32>, // Shape: [LABELS, IMAGE_DIM]
    biases: Array1<f32>,  // Shape: [LABELS]
}

impl Model for LinearModel {
    fn new() -> anyhow::Result<Self> {
        let mut rng = rand::thread_rng();
        
        // Initialize weights with small random values (Xavier initialization)
        let scale = (2.0 / IMAGE_DIM as f32).sqrt();
        let weights = Array2::from_shape_fn((LABELS, IMAGE_DIM), |_| {
            rng.gen::<f32>() * scale - scale / 2.0
        });
        
        // Initialize biases to zero
        let biases = Array1::zeros(LABELS);
        
        Ok(Self { weights, biases })
    }

    fn forward(&self, xs: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
        // xs shape: [batch_size, IMAGE_DIM]
        // weights shape: [LABELS, IMAGE_DIM]
        // result shape: [batch_size, LABELS]
        
        let logits = xs.dot(&self.weights.t()) + &self.biases;
        Ok(logits)
    }

    fn set_parameters(&mut self, params: &ModelParameters) -> anyhow::Result<()> {
        // Reshape flat weights back to 2D array
        if params.weights.len() != LABELS * IMAGE_DIM {
            return Err(anyhow::anyhow!("Invalid weight dimensions"));
        }
        if params.biases.len() != LABELS {
            return Err(anyhow::anyhow!("Invalid bias dimensions"));
        }
        
        let weights_2d = Array2::from_shape_vec((LABELS, IMAGE_DIM), params.weights.clone())?;
        let biases_1d = Array1::from_vec(params.biases.clone());
        
        self.weights = weights_2d;
        self.biases = biases_1d;
        
        Ok(())
    }

    fn get_parameters(&self) -> anyhow::Result<ModelParameters> {
        let weights = if self.weights.is_standard_layout() {
            self.weights.as_slice().unwrap().to_vec()
        } else {
            self.weights.iter().cloned().collect()
        };
        
        let biases = if self.biases.is_standard_layout() {
            self.biases.as_slice().unwrap().to_vec()
        } else {
            self.biases.iter().cloned().collect()
        };
        
        Ok(ModelParameters {
            weights,
            biases,
            weight_shape: vec![LABELS as i32, IMAGE_DIM as i32],
            bias_shape: vec![LABELS as i32],
        })
    }

    fn train(&mut self, x: &Array2<f32>, y: &Array1<i32>, learning_rate: f32, epochs: usize) -> anyhow::Result<()> {
        let batch_size = x.shape()[0];
        
        for epoch in 0..epochs {
            // Forward pass
            let logits = self.forward(x)?;
            
            // Softmax and cross-entropy loss
            let softmax = softmax(&logits)?;
            let loss = cross_entropy_loss(&softmax, y)?;
            
            // Backward pass - compute gradients
            let grad_output = softmax_cross_entropy_gradient(&softmax, y)?;
            
            // Gradient w.r.t. weights: X^T * grad_output
            let grad_weights = x.t().dot(&grad_output);
            
            // Gradient w.r.t. biases: sum over batch dimension
            let grad_biases = grad_output.sum_axis(ndarray::Axis(0));
            
            // Update parameters
            self.weights = &self.weights - learning_rate / batch_size as f32 * &grad_weights.t();
            self.biases = &self.biases - learning_rate / batch_size as f32 * &grad_biases;
            
            if epoch % 10 == 0 {
                let accuracy = compute_accuracy(&logits, y)?;
                log::info!("Epoch {}: loss = {:.4}, accuracy = {:.2}%", epoch, loss, accuracy * 100.0);
            }
        }
        
        Ok(())
    }
}

// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingArgs {
    pub learning_rate: f64,
    pub epochs: usize,
}

// Federated averaging implementation
pub fn federate_average(client_params: Vec<ModelParameters>) -> anyhow::Result<ModelParameters> {
    if client_params.is_empty() {
        return Err(anyhow::anyhow!("No client parameters to average"));
    }

    let n_clients = client_params.len() as f32;
    let first_params = &client_params[0];
    
    // Initialize averaged parameters with zeros
    let mut avg_weights = vec![0.0; first_params.weights.len()];
    let mut avg_biases = vec![0.0; first_params.biases.len()];
    
    // Sum all client parameters
    for params in &client_params {
        for (i, &weight) in params.weights.iter().enumerate() {
            avg_weights[i] += weight;
        }
        for (i, &bias) in params.biases.iter().enumerate() {
            avg_biases[i] += bias;
        }
    }
    
    // Average the parameters
    for weight in &mut avg_weights {
        *weight /= n_clients;
    }
    for bias in &mut avg_biases {
        *bias /= n_clients;
    }
    
    Ok(ModelParameters {
        weights: avg_weights,
        biases: avg_biases,
        weight_shape: first_params.weight_shape.clone(),
        bias_shape: first_params.bias_shape.clone(),
    })
}

// Neural network helper functions
pub fn softmax(logits: &Array2<f32>) -> anyhow::Result<Array2<f32>> {
    let mut result = logits.clone();
    
    for mut row in result.rows_mut() {
        let max_val = row.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        row.mapv_inplace(|x| (x - max_val).exp());
        let sum = row.sum();
        row.mapv_inplace(|x| x / sum);
    }
    
    Ok(result)
}

pub fn cross_entropy_loss(softmax_probs: &Array2<f32>, labels: &Array1<i32>) -> anyhow::Result<f32> {
    let batch_size = softmax_probs.shape()[0];
    let mut loss = 0.0;
    
    for (i, &label) in labels.iter().enumerate() {
        let prob = softmax_probs[[i, label as usize]];
        loss -= prob.max(1e-15).ln(); // Add small epsilon to prevent log(0)
    }
    
    Ok(loss / batch_size as f32)
}

pub fn softmax_cross_entropy_gradient(softmax_probs: &Array2<f32>, labels: &Array1<i32>) -> anyhow::Result<Array2<f32>> {
    let mut grad = softmax_probs.clone();
    
    for (i, &label) in labels.iter().enumerate() {
        grad[[i, label as usize]] -= 1.0;
    }
    
    Ok(grad)
}

pub fn compute_accuracy(logits: &Array2<f32>, labels: &Array1<i32>) -> anyhow::Result<f32> {
    let batch_size = logits.shape()[0];
    let mut correct = 0;
    
    for (i, &true_label) in labels.iter().enumerate() {
        let predicted_label = logits.row(i)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0 as i32;
            
        if predicted_label == true_label {
            correct += 1;
        }
    }
    
    Ok(correct as f32 / batch_size as f32)
}

// Calculate model accuracy (public interface)
pub fn calculate_accuracy<M: Model>(
    model: &M,
    test_images: &Array2<f32>,
    test_labels: &Array1<i32>,
) -> anyhow::Result<f32> {
    let logits = model.forward(test_images)?;
    compute_accuracy(&logits, test_labels)
}
