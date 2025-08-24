use smollu_emulator::{SmolluEmulator, SmolluEmulatorApp, VmError};
use eframe::egui;
use std::env;
use std::path::PathBuf;
use std::process;

fn main() {
    env_logger::init();

    let args: Vec<String> = env::args().collect();

    // Check for CLI mode flag
    let cli_mode = args.contains(&"--cli".to_string());

    if cli_mode {
        run_cli_mode(&args);
    } else {
        run_gui_mode(&args);
    }
}

fn run_gui_mode(args: &[String]) {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1000.0, 700.0]),
        ..Default::default()
    };

    let app = if args.len() >= 2 && !args[1].starts_with("--") {
        let file_path = PathBuf::from(&args[1]);
        if file_path.exists() {
            SmolluEmulatorApp::with_file(file_path)
        } else {
            eprintln!("Warning: File '{}' does not exist. Starting with empty emulator.", args[1]);
            SmolluEmulatorApp::new()
        }
    } else {
        SmolluEmulatorApp::new()
    };

    if let Err(e) = eframe::run_native(
        "Smollu VM Emulator",
        options,
        Box::new(|_cc| Box::new(app)),
    ) {
        eprintln!("Failed to run GUI: {}", e);
        process::exit(1);
    }
}

fn run_cli_mode(args: &[String]) {
    if args.len() < 3 {
        eprintln!("Usage: {} --cli <bytecode_file>", args[0]);
        process::exit(1);
    }

    let bytecode_file = &args[2];

    // Create emulator and load bytecode
    let mut emulator = match SmolluEmulator::new() {
        Ok(emu) => emu,
        Err(e) => {
            eprintln!("Failed to create emulator: {}", e);
            process::exit(1);
        }
    };

    if let Err(e) = emulator.load_bytecode_file(bytecode_file) {
        eprintln!("Failed to load bytecode file '{}': {}", bytecode_file, e);
        process::exit(1);
    }

    println!("Loaded bytecode from: {}", bytecode_file);
    println!("Starting VM execution...\n");

    // Debug: check initial VM state
    let initial_state = emulator.get_vm_state();
    println!(
        "Initial state - PC: {}, SP: {}",
        initial_state.pc, initial_state.sp
    );

    // Run the VM
    match emulator.run() {
        Ok(exit_code) => {
            println!("\nVM execution completed with exit code: {}", exit_code);

            // Display any captured output
            let output_history = emulator.get_output_history();
            if !output_history.is_empty() {
                println!("\nCaptured output:");
                for (i, output) in output_history.iter().enumerate() {
                    println!("[{}] {}", i + 1, output);
                }
            }

            // Display final VM state
            let state = emulator.get_vm_state();
            println!("\nFinal VM state:");
            println!("  PC: {}", state.pc);
            println!("  Stack size: {}", state.sp);
            if let Some(top_value) = state.stack_top {
                println!("  Stack top: {}", top_value);
            }
        }
        Err(VmError::ExecutionError(code)) => {
            let state = emulator.get_vm_state();
            eprintln!("VM execution failed with error code: {}", code);
            eprintln!("  PC: {}", state.pc);
            eprintln!("  Stack size: {}", state.sp);
            process::exit(1);
        }
        Err(e) => {
            eprintln!("VM execution failed: {}", e);
            let state = emulator.get_vm_state();
            eprintln!("  PC: {}", state.pc);
            eprintln!("  Stack size: {}", state.sp);
            process::exit(1);
        }
    }
}
