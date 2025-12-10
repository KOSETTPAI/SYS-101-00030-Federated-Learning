use anyhow::{anyhow, Result};
use ndarray::{Array1, Array2};
use log::info;
use byteorder::{BigEndian, ReadBytesExt};
use flate2::read::GzDecoder;
use std::io::Read;
use std::path::PathBuf;
use tokio::fs;
use tokio::io::AsyncWriteExt;

pub struct MnistData {
    pub train_images: Array2<f32>,
    pub train_labels: Array1<i32>,
    pub test_images: Array2<f32>,
    pub test_labels: Array1<i32>,
}

impl MnistData {
    pub async fn load() -> Result<Self> {
        // Prefer https mirror for reliability; fall back to Yann LeCun host if needed.
        const BASE_URLS: [&str; 2] = [
            "https://storage.googleapis.com/cvdf-datasets/mnist",
            "http://yann.lecun.com/exdb/mnist",
        ];
        const TRAIN_IMAGES: &str = "train-images-idx3-ubyte.gz";
        const TRAIN_LABELS: &str = "train-labels-idx1-ubyte.gz";
        const TEST_IMAGES: &str = "t10k-images-idx3-ubyte.gz";
        const TEST_LABELS: &str = "t10k-labels-idx1-ubyte.gz";

        let data_dir = PathBuf::from("data/mnist");
        fs::create_dir_all(&data_dir).await?;

        // Download + inflate with one automatic retry on corruption
        async fn fetch_and_inflate(urls: &[&str], filename: &str, path: &PathBuf) -> Result<Vec<u8>> {
            async fn download(url: &str, path: &PathBuf) -> Result<()> {
                let bytes = reqwest::get(url).await?.bytes().await?;
                let mut file = fs::File::create(path).await?;
                file.write_all(&bytes).await?;
                Ok(())
            }

            async fn inflate(path: &PathBuf) -> Result<Vec<u8>> {
                let bytes = fs::read(path).await?;
                // Basic gzip header check (1F 8B)
                if bytes.len() < 2 || bytes[0] != 0x1f || bytes[1] != 0x8b {
                    return Err(anyhow!("Invalid gzip header"));
                }
                let mut decoder = GzDecoder::new(&bytes[..]);
                let mut out = Vec::new();
                decoder.read_to_end(&mut out)?;
                Ok(out)
            }

            // Try each URL, with one retry per URL on corruption
            for &base in urls {
                let url = format!("{}/{}", base, filename);
                for attempt in 0..2 {
                    // Always re-download on second attempt or if file missing
                    if attempt == 1 || !path.exists() {
                        let _ = fs::remove_file(path).await; // remove any bad file
                        download(&url, path).await?;
                    }
                    match inflate(path).await {
                        Ok(data) => return Ok(data),
                        Err(e) => {
                            // retry once per URL; then move to next URL
                            if attempt == 0 {
                                continue;
                            } else {
                                let _ = fs::remove_file(path).await;
                                break;
                            }
                        }
                    }
                }
            }

            Err(anyhow!("Failed to download or inflate {}", filename))
        }

        let train_images_path = data_dir.join(TRAIN_IMAGES);
        let train_labels_path = data_dir.join(TRAIN_LABELS);
        let test_images_path = data_dir.join(TEST_IMAGES);
        let test_labels_path = data_dir.join(TEST_LABELS);

        let (train_images_raw, train_labels_raw, test_images_raw, test_labels_raw) = tokio::try_join!(
            fetch_and_inflate(&BASE_URLS, TRAIN_IMAGES, &train_images_path),
            fetch_and_inflate(&BASE_URLS, TRAIN_LABELS, &train_labels_path),
            fetch_and_inflate(&BASE_URLS, TEST_IMAGES, &test_images_path),
            fetch_and_inflate(&BASE_URLS, TEST_LABELS, &test_labels_path),
        )?;

        fn parse_images(bytes: &[u8]) -> Result<Array2<f32>> {
            fn read_u32_be(rdr: &mut std::io::Cursor<&[u8]>) -> Result<u32> {
                Ok(ReadBytesExt::read_u32::<BigEndian>(rdr)?)
            }
            fn read_u32_be_usize(rdr: &mut std::io::Cursor<&[u8]>) -> Result<usize> {
                Ok(read_u32_be(rdr)? as usize)
            }

            let mut rdr = std::io::Cursor::new(bytes);
            let magic = read_u32_be(&mut rdr)?;
            if magic != 2051 {
                return Err(anyhow!("Invalid magic for images: {}", magic));
            }
            let num = read_u32_be_usize(&mut rdr)?;
            let rows = read_u32_be_usize(&mut rdr)?;
            let cols = read_u32_be_usize(&mut rdr)?;
            let expected = rows * cols * num;
            let mut data = vec![0u8; expected];
            rdr.read_exact(&mut data)?;
            let images = Array2::from_shape_vec(
                (num, rows * cols),
                data.into_iter().map(|v| v as f32 / 255.0).collect(),
            )?;
            Ok(images)
        }

        fn parse_labels(bytes: &[u8]) -> Result<Array1<i32>> {
            fn read_u32_be(rdr: &mut std::io::Cursor<&[u8]>) -> Result<u32> {
                Ok(ReadBytesExt::read_u32::<BigEndian>(rdr)?)
            }
            fn read_u32_be_usize(rdr: &mut std::io::Cursor<&[u8]>) -> Result<usize> {
                Ok(read_u32_be(rdr)? as usize)
            }

            let mut rdr = std::io::Cursor::new(bytes);
            let magic = read_u32_be(&mut rdr)?;
            if magic != 2049 {
                return Err(anyhow!("Invalid magic for labels: {}", magic));
            }
            let num = read_u32_be_usize(&mut rdr)?;
            let mut data = vec![0u8; num];
            rdr.read_exact(&mut data)?;
            Ok(Array1::from_iter(data.into_iter().map(|v| v as i32)))
        }

        let train_images = parse_images(&train_images_raw)?;
        let train_labels = parse_labels(&train_labels_raw)?;
        let test_images = parse_images(&test_images_raw)?;
        let test_labels = parse_labels(&test_labels_raw)?;

        info!("Loaded MNIST dataset:");
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
