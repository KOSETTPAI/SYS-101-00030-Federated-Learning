fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto/federated_learning.proto");
    println!("cargo:rerun-if-changed=proto");

    // Use a vendored protoc so contributors don't need a system install
    let protoc_path = protoc_bin_vendored::protoc_bin_path()?;
    std::env::set_var("PROTOC", protoc_path);

    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .compile(&["proto/federated_learning.proto"], &["proto"])?;

    Ok(())
}
