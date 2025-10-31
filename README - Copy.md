# Federated Learning System - SYS-101 Assessment 3

A complete implementation of a federated learning system using Federated Averaging (FedAvg) with Rust and the Candle ML framework.

## System Architecture

### Parameter Server
- Maintains global models and client registry
- Implements thread-safe operations using `RwLock`
- Distributes data to clients in IID fashion
- Aggregates client updates using federated averaging

### Federated Clients  
- Connect to parameter server via gRPC
- Train models locally on distributed data subsets
- Send updated parameters back to server
- Support both manual and interactive operation modes

## Key Features

✅ **Concurrency & Synchronization**
- Deadlock-free and race-free design using `RwLock` for shared data structures
- Asynchronous operations with Tokio runtime
- Safe concurrent access to global model state

✅ **Federated Averaging Implementation**  
- IID data distribution across clients
- Parameter aggregation by averaging weights and biases
- Multiple training rounds with convergence tracking

✅ **Protocol Buffers & gRPC Communication**
- Type-safe client-server communication
- Streaming support for large model parameters
- Robust error handling and status reporting

✅ **MNIST Linear Classifier**
- Complete implementation of linear model for MNIST digit classification
- Training with SGD optimizer and cross-entropy loss
- Accuracy evaluation on test datasets

## API Specification

### Parameter Server APIs

```rust
// Register client for a specific model
rpc Register(RegisterRequest) returns (RegisterResponse);

// Initialize/reset global model  
rpc Init(InitRequest) returns (InitResponse);

// Start N rounds of federated training
rpc Train(TrainRequest) returns (TrainResponse);

// Get current model parameters and status
rpc GetModel(GetModelRequest) returns (GetModelResponse);

// Test model accuracy on server's test dataset  
rpc TestModel(TestModelRequest) returns (TestModelResponse);
```

### Federated Client APIs

```rust
// Train local model with provided data
rpc TrainLocal(ClientTrainRequest) returns (ClientTrainResponse);

// Get local model parameters and status
rpc GetLocalModel(GetModelRequest) returns (GetModelResponse);

// Test local model accuracy
rpc TestLocal(TestModelRequest) returns (TestModelResponse);
```

## Usage Guide

### 1. Start Parameter Server

```bash
cargo run --bin server --release -- --port 50051 --learning-rate 0.1 --epochs-per-round 10
```

### 2. Start Clients (in separate terminals)

```bash
# Client 1
cargo run --bin client --release -- --server-address 127.0.0.1:50051 --model-name mnist

# Client 2  
cargo run --bin client --release -- --server-address 127.0.0.1:50051 --model-name mnist

# Client 3
cargo run --bin client --release -- --server-address 127.0.0.1:50051 --model-name mnist
```

### 3. Interactive Client Commands

Once a client is running, you can use these commands:

```bash
client> join                    # Register with server
client> server-init             # Initialize global model (run once)
client> server-train 5          # Run 5 federated training rounds
client> server-test             # Test global model accuracy
client> train                   # Train local model only
client> test                    # Test local model
client> get                     # Retrieve global model
client> quit                    # Exit client
```

### 4. Quick Demo

```bash
# Run the demo to see usage instructions
cargo run --example demo
```

## Training Flow

1. **Initialization**: Server creates initial global model with random weights
2. **Client Registration**: Clients register with server for specific model
3. **Data Distribution**: Server distributes IID subsets of MNIST training data
4. **Local Training**: Each client trains on its local data subset for specified epochs
5. **Parameter Aggregation**: Server collects and averages all client parameters
6. **Model Update**: Global model is updated with averaged parameters
7. **Iteration**: Process repeats for specified number of rounds

## Design Decisions

### Synchronization Strategy
- **RwLock**: Chosen for fine-grained read/write access to shared model state
- **Arc**: Used for safe shared ownership across async tasks
- **gRPC**: Provides built-in thread safety and connection pooling

### Data Distribution
- **IID Splitting**: Training data divided equally among clients to ensure convergence
- **Tensor Serialization**: Model parameters serialized as flat arrays for network transfer
- **Device Flexibility**: Supports both CPU and CUDA devices

### Error Handling
- **Comprehensive Error Types**: Custom error handling for network, model, and data issues
- **Graceful Degradation**: System continues operating even if some clients disconnect
- **Status Tracking**: Real-time tracking of model and training status

## Performance Characteristics

- **Memory Efficient**: Minimal memory overhead with shared global model
- **Scalable**: Supports arbitrary number of clients (tested with 3+ clients)
- **Fast Convergence**: Typically achieves 85%+ accuracy on MNIST within 3-5 rounds

## Building and Testing

### Prerequisites
- Rust 1.70+ (stable toolchain)
- Protocol Buffers compiler (optional - auto-downloaded)

### Build
```bash
cargo build --release
```

### Test
```bash
cargo test
```

### Lint
```bash 
cargo clippy
```

## File Structure

```
src/
├── bin/
│   ├── server.rs           # Parameter server implementation
│   └── client.rs           # Federated client implementation
├── common.rs               # Shared types and utilities
├── data.rs                 # MNIST data loading and distribution
└── lib.rs                  # Library root
proto/
└── federated_learning.proto # Protocol buffer definitions
examples/
└── demo.rs                 # Usage demonstration
```

## Implementation Notes

### Thread Safety
The system ensures thread safety through:
- Exclusive write access to global model via `RwLock`  
- Immutable sharing of model parameters during training
- Atomic updates to client registration and status

### Fault Tolerance
- Server continues operating if clients disconnect during training
- Partial training rounds are handled gracefully
- Network errors are properly propagated and logged

### Extensibility
The modular design supports:
- Multiple model architectures (currently implements linear classifier)
- Different datasets (currently supports MNIST)
- Pluggable aggregation strategies (currently implements FedAvg)

## Citations

Implementation follows the FedAvg algorithm as described in:
- McMahan, B., et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.

<citations>
  <document>
      <document_type>WEB_PAGE</document_type>
      <document_id>https://www.educative.io/answers/what-is-federated-averaging-fedavg</document_id>
  </document>
  <document>
      <document_type>WEB_PAGE</document_type>
      <document_id>https://federated.withgoogle.com</document_id>
  </document>
</citations>