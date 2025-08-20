use smollu_emulator::SmolluEmulator;
use std::fs;

fn main() -> anyhow::Result<()> {
    println!("=== Simple Emulator Test ===");

    // Create emulator
    let mut emulator = SmolluEmulator::new()?;
    println!("✓ Created emulator");

    // Try to load the bytecode file and inspect it first
    let bytecode_path = "../build/demo/Simple demo/demo.smolbc";

    match fs::read(bytecode_path) {
        Ok(bytecode) => {
            println!("✓ Read {} bytes from {}", bytecode.len(), bytecode_path);
            println!("  Header: {:02x?}", &bytecode[0..16.min(bytecode.len())]);

            // Try to load it
            match emulator.load_bytecode(&bytecode) {
                Ok(()) => {
                    println!("✓ Successfully loaded bytecode");

                    // Check VM state after loading
                    let state = emulator.get_vm_state();
                    println!("  Initial state - PC: {}, SP: {}", state.pc, state.sp);

                    // Try to run
                    println!("\n--- Attempting to run VM ---");
                    match emulator.run() {
                        Ok(exit_code) => {
                            println!("✓ VM completed with exit code: {}", exit_code);

                            // Check output
                            let output = emulator.get_output_history();
                            if output.is_empty() {
                                println!("⚠ No output captured");
                            } else {
                                println!("✓ Captured {} lines of output:", output.len());
                                for line in output {
                                    println!("  {}", line);
                                }
                            }

                            // Check final state
                            let final_state = emulator.get_vm_state();
                            println!(
                                "  Final state - PC: {}, SP: {}",
                                final_state.pc, final_state.sp
                            );
                        }
                        Err(e) => {
                            println!("✗ VM execution failed: {}", e);
                            return Err(e.into());
                        }
                    }
                }
                Err(e) => {
                    println!("✗ Failed to load bytecode: {}", e);
                    return Err(e.into());
                }
            }
        }
        Err(e) => {
            println!("✗ Failed to read bytecode file: {}", e);
            println!("  Make sure you've built the C project first:");
            println!("  cd ..; mkdir build; cd build; cmake .. -DBUILD_TESTS=OFF; cmake --build .");
            println!("  cd 'demo/Simple demo'; ./smollu_compiler demo.smol -o demo.smolbc");
            return Ok(());
        }
    }

    Ok(())
}
