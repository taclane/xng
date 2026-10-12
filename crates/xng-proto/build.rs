fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=../../proto/asf2.proto");
    // Parse the schema with protox (a pure-Rust protobuf compiler) so the
    // build needs no system `protoc`.
    let fds = protox::compile(["asf2.proto"], ["../../proto"])?;
    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .compile_fds(fds)?;
    Ok(())
}
