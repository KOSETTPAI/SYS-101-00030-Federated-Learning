use anyhow::Result;
use ndarray::{Array1, Array2};
use log::info;
use rand::Rng;

pub struct MnistData {
    pub train_images: Array2<f32>,
    pub train_labels: Array1<i32>,
    pub test_images: Array2<f32>,
    pub test_labels: Array1<i32>,
}

impl MnistData {
    pub async fn load() -> Result<Self> {
        // Create synthetic MNIST-like data for demonstration
        // In a real implementation, you would download and parse actual MNIST files
        
        const TRAIN_SIZE: usize = 1000;  // Reduced size for demo
        const TEST_SIZE: usize = 200;
        const IMAGE_DIM: usize = 784;     // 28x28 pixels
        const NUM_CLASSES: usize = 10;
        
        let mut rng = rand::thread_rng();
        use rand::Rng;
        
        // Generate synthetic training data
        let train_images = Array2::from_shape_fn((TRAIN_SIZE, IMAGE_DIM), |_| {
            rng.gen::<f32>()
        });
        
        let train_labels = Array1::from_shape_fn(TRAIN_SIZE, |_| {
            rng.gen_range(0..NUM_CLASSES) as i32
        });
        
        // Generate synthetic test data
        let test_images = Array2::from_shape_fn((TEST_SIZE, IMAGE_DIM), |_| {
            rng.gen::<f32>()
        });
        
        let test_labels = Array1::from_shape_fn(TEST_SIZE, |_| {
            rng.gen_range(0..NUM_CLASSES) as i32
        });
        
        info!("Loaded synthetic MNIST-like dataset:");
        info!("  Train images: {:?}", train_images.shape());
        info!("  Train labels: {:?}", train_labels.shape());
        info!("  Test images: {:?}", test_images.shape());
        info!("  Test labels: {:?}", test_labels.shape());
        
        Ok(Self {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }
    
    /// Split training data into IID subsets for federated clients
    pub fn split_train_data(&self, num_clients: usize) -> Result<Vec<(Array2<f32>, Array1<i32>)>> {
        let total_samples = self.train_images.shape()[0];
        let samples_per_client = total_samples / num_clients;
        
        let mut client_data = Vec::new();
        
        for i in 0..num_clients {
            let start_idx = i * samples_per_client;
            let end_idx = if i == num_clients - 1 {
                total_samples // Last client gets remaining samples
            } else {
                (i + 1) * samples_per_client
            };
            
            let client_images = self.train_images.slice(ndarray::s![start_idx..end_idx, ..]).to_owned();
            let client_labels = self.train_labels.slice(ndarray::s![start_idx..end_idx]).to_owned();
            
            client_data.push((client_images, client_labels));
            
            info!("Client {}: {} samples ({}-{})", i, end_idx - start_idx, start_idx, end_idx);
        }
        
        Ok(client_data)
    }
    
    /// Convert training data to flat vectors for network transmission
    pub fn train_data_to_flat(&self) -> Result<(Vec<f32>, Vec<i32>)> {
        let images_flat = self.train_images.as_slice().unwrap().to_vec();
        let labels_flat = self.train_labels.as_slice().unwrap().to_vec();
        
        Ok((images_flat, labels_flat))
    }
    
    /// Convert test data to flat vectors for network transmission
    pub fn test_data_to_flat(&self) -> Result<(Vec<f32>, Vec<i32>)> {
        let images_flat = self.test_images.as_slice().unwrap().to_vec();
        let labels_flat = self.test_labels.as_slice().unwrap().to_vec();
        
        Ok((images_flat, labels_flat))
    }
}

/// Create arrays from flat data (useful for client-side reconstruction)
pub fn arrays_from_flat(
    images_flat: Vec<f32>, 
    labels_flat: Vec<i32>
) -> Result<(Array2<f32>, Array1<i32>)> {
    let num_samples = labels_flat.len();
    let image_dim = images_flat.len() / num_samples;
    
    let images = Array2::from_shape_vec((num_samples, image_dim), images_flat)?;
    let labels = Array1::from_vec(labels_flat);
    
    Ok((images, labels))
}
