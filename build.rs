fn main() {
    // Using manual protobuf implementation, no build script needed
    println!("cargo:rerun-if-changed=proto/federated_learning.proto");
}
