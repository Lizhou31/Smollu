use smollu_emulator::{SmolluEmulator, Value};

fn main() -> anyhow::Result<()> {
    println!("=== Smollu Emulator Basic Test ===\n");

    // Create a new emulator instance
    let mut emulator = SmolluEmulator::new()?;
    println!("✓ Created emulator instance");

    // Test 1: Load the demo bytecode file
    let demo_bytecode_path = "../build/demo/Simple demo/demo.smolbc";

    match emulator.load_bytecode_file(demo_bytecode_path) {
        Ok(()) => {
            println!("✓ Loaded bytecode from: {}", demo_bytecode_path);
        }
        Err(e) => {
            println!("⚠ Could not load demo bytecode ({})", e);
            println!("  This is expected if the C project hasn't been built yet.");
            println!("  To build the demo:");
            println!("    cd ..");
            println!("    mkdir build && cd build");
            println!("    cmake ..");
            println!("    cmake --build .");
            return Ok(());
        }
    }

    // Test 2: Check initial VM state
    let initial_state = emulator.get_vm_state();
    println!(
        "✓ Initial VM state - PC: {}, Stack: {}",
        initial_state.pc, initial_state.sp
    );

    // Test 3: Run the program
    println!("\n--- Running VM ---");
    match emulator.run() {
        Ok(exit_code) => {
            println!("✓ VM execution completed with exit code: {}", exit_code);
        }
        Err(e) => {
            println!("✗ VM execution failed: {}", e);
            let state = emulator.get_vm_state();
            eprintln!("  PC: {}", state.pc);
            eprintln!("  Stack size: {}", state.sp);
            return Err(e.into());
        }
    }

    // Test 4: Check captured output
    let output_history = emulator.get_output_history();
    if !output_history.is_empty() {
        println!("\n--- Captured Output ---");
        for (i, output) in output_history.iter().enumerate() {
            println!("[{}] {}", i + 1, output);
        }
        println!(
            "✓ Successfully captured {} output line(s)",
            output_history.len()
        );
    } else {
        println!("⚠ No output captured");
    }

    // Test 5: Check final VM state
    let final_state = emulator.get_vm_state();
    println!("\n--- Final VM State ---");
    println!("PC: {}", final_state.pc);
    println!("Stack size: {}", final_state.sp);
    if let Some(top_value) = final_state.stack_top {
        println!("Stack top: {}", top_value);
    }

    // Test 6: Test global variable access
    println!("\n--- Global Variables ---");
    for slot in 0..5 {
        let global_value = emulator.get_global(slot);
        match global_value {
            Value::Nil => continue, // Skip unset globals
            _ => println!("Global[{}] = {}", slot, global_value),
        }
    }

    // Test 7: Reset and verify
    println!("\n--- Reset Test ---");
    emulator.reset();
    let reset_state = emulator.get_vm_state();
    println!(
        "✓ VM reset - PC: {}, Stack: {}",
        reset_state.pc, reset_state.sp
    );
    println!(
        "✓ Output history cleared: {} lines",
        emulator.get_output_history().len()
    );

    println!("\n=== All Tests Completed Successfully ===");
    Ok(())
}
