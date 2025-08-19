use std::env;
use std::path::PathBuf;

fn main() {
    // Get the directory where this build script is located (emulator/)
    let current_dir = env::current_dir().expect("Failed to get current directory");
    
    // Parent directory should be the Smollu root
    let smollu_root = current_dir.parent().expect("Failed to get parent directory");
    
    // Paths to the original C source files
    let vm_src = smollu_root.join("src/components/vm/smollu_vm.c");
    let compiler_src = smollu_root.join("src/components/compiler");
    
    // Tell cargo to watch for changes in these files
    println!("cargo:rerun-if-changed={}", vm_src.display());
    println!("cargo:rerun-if-changed={}", compiler_src.display());
    println!("cargo:rerun-if-changed=c_integration/wrapper.c");
    println!("cargo:rerun-if-changed=c_integration/wrapper.h");
    
    // Compile the C VM and wrapper code
    cc::Build::new()
        .file(vm_src)
        .file("c_integration/wrapper.c")
        .include(smollu_root.join("src/components/vm"))
        .include(smollu_root.join("src/components/compiler"))
        .include("c_integration")
        .compile("smollu_vm");
    
    // Generate bindings using bindgen
    let bindings = bindgen::Builder::default()
        // Header file to generate bindings for
        .header("c_integration/wrapper.h")
        // Include paths
        .clang_arg(format!("-I{}", smollu_root.join("src/components/vm").display()))
        .clang_arg(format!("-I{}", smollu_root.join("src/components/compiler").display()))
        .clang_arg("-Ic_integration")
        // Generate bindings for these functions and types
        .allowlist_function("smollu_.*")
        .allowlist_function("wrapper_.*")
        .allowlist_type("SmolluVM")
        .allowlist_type("Value")
        .allowlist_type("ValueType")
        .allowlist_type("NativeFn")
        .allowlist_var("VAL_.*")
        .allowlist_var("SMOLLU_.*")
        // Generate the bindings
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("Unable to generate bindings");
    
    // Write the bindings to the output file
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}