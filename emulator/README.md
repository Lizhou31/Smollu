# Smollu Emulator

A Rust-based emulator for the Smollu VM with GUI capabilities for hardware simulation. This emulator integrates the existing C VM core and provides additional debugging and simulation features.

## Overview

The Smollu Emulator is designed to provide a modern development environment for Smollu programs with:

- **Safe Rust Integration**: FFI bindings over the C VM with memory safety
- **Hardware Simulation**: Virtual LED matrices, GPIO pins, and sensors (planned)
- **GUI Interface**: Real-time visualization and debugging tools (planned)
- **Enhanced Debugging**: VM state inspection and execution control

## Project Status

- ✅ **Phase 1 Complete**: Core C VM Integration
- 🚧 **Phase 2 Planned**: Basic GUI Framework
- 📋 **Phase 3 Planned**: Hardware Simulation
- 📋 **Phase 4 Planned**: Advanced Features
- 📋 **Phase 5 Planned**: Polish & Examples

## Quick Start

### Prerequisites

- Rust 1.70+ with Cargo
- CMake 3.15+
- C compiler (GCC/Clang)
- The parent Smollu project built

### Building

1. **Build the C VM first** (from project root):
   ```bash
   cd ..
   mkdir build && cd build
   cmake .. -DBUILD_TESTS=OFF
   cmake --build .
   ```

2. **Create demo bytecode**:
   ```bash
   cd "demo/Simple demo"
   ./smollu_compiler demo.smol -o demo.smolbc
   ```

3. **Build the emulator**:
   ```bash
   cd ../../emulator
   cargo build
   ```

### Running

Run the emulator with a bytecode file:

```bash
cargo run -- "../build/demo/Simple demo/demo.smolbc"
```

Run the basic integration test:

```bash
cargo run --example basic_test
```

Run the simple debugging test:

```bash
cargo run --example simple_test
```

## Architecture

### Project Structure

```
emulator/
├── Cargo.toml              # Rust project configuration
├── build.rs                # C integration build script
├── README.md               # This file
├── src/
│   ├── main.rs             # Command-line interface
│   ├── lib.rs              # Main library interface
│   └── vm/                 # VM integration layer
│       ├── mod.rs          # Module exports
│       ├── bindings.rs     # Generated C bindings
│       └── wrapper.rs      # Safe Rust wrappers
├── c_integration/          # C wrapper layer
│   ├── wrapper.h           # C API declarations
│   └── wrapper.c           # C API implementations
└── examples/               # Test programs
    ├── basic_test.rs       # Comprehensive integration test
    └── simple_test.rs      # Simple debugging test
```

### Core Components

#### 1. **C Integration Layer** (`c_integration/`)
- **wrapper.h/c**: Simplified C API over the Smollu VM
- **Safe FFI**: Memory management and error handling
- **Enhanced Natives**: Emulator-specific native functions

#### 2. **Rust VM Module** (`src/vm/`)
- **bindings.rs**: Auto-generated FFI bindings via `bindgen`
- **wrapper.rs**: Safe Rust abstractions with proper error handling
- **Type Safety**: Rust enums and structs for VM values and states

#### 3. **Emulator Library** (`src/lib.rs`)
- **High-level API**: `SmolluEmulator` struct for easy integration
- **Output Capture**: History tracking for print statements
- **Debug Interface**: VM state inspection and control

#### 4. **Command-line Interface** (`src/main.rs`)
- **File Loading**: Bytecode file handling
- **Execution Control**: Run, debug, and inspect VM state
- **Output Display**: Formatted results and debugging info

## Usage Examples

### Basic Emulation

```rust
use smollu_emulator::{SmolluEmulator, Value};

// Create emulator
let mut emulator = SmolluEmulator::new()?;

// Load bytecode
emulator.load_bytecode_file("program.smolbc")?;

// Run program
let exit_code = emulator.run()?;

// Get output
let output = emulator.get_output_history();
for line in output {
    println!("Output: {}", line);
}
```

### VM State Inspection

```rust
// Check VM state
let state = emulator.get_vm_state();
println!("PC: {}, Stack: {}", state.pc, state.sp);

// Access globals
let global_var = emulator.get_global(0);
println!("Global[0] = {}", global_var);

// Modify globals
emulator.set_global(1, Value::Int(42));
```

### Debug and Reset

```rust
// Reset VM state
emulator.reset();

// Clear output history
emulator.clear_output_history();

// Run again
emulator.run()?;
```

## Development

### Building with Debug Info

```bash
cargo build
RUST_LOG=debug cargo run -- bytecode_file.smolbc
```

### Running Tests

```bash
cargo test
cargo run --example basic_test
cargo run --example simple_test
```

### Adding New Features

1. **Native Functions**: Add to `c_integration/wrapper.c`
2. **Rust API**: Extend `src/vm/wrapper.rs`
3. **High-level Interface**: Update `src/lib.rs`

## Troubleshooting

### Build Issues

**"cmake not found"**:
```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt install cmake

# Windows
# Install CMake from https://cmake.org/
```

**"bindgen failed"**:
- Ensure C compiler is available
- Check include paths in `build.rs`
- Verify parent project builds successfully

### Runtime Issues

**"Failed to load bytecode"**:
- Ensure the C project is built first
- Check bytecode file exists and has correct format
- Verify file permissions

**"No output captured"**:
- VM might be failing silently
- Check VM execution with `RUST_LOG=debug`
- Verify bytecode is compatible

## Roadmap

### Phase 2: GUI Framework (Planned)
- egui integration for immediate-mode GUI
- Basic application window with file loading
- Console output widget for VM output
- VM execution controls (play/pause/reset)

### Phase 3: Hardware Simulation (Planned)
- LED matrix simulation (8x8 default, configurable)
- GPIO pin visualization and control
- Native functions for hardware interaction
- Real-time state updates

### Phase 4: Advanced Features (Planned)
- Breakpoint debugging and step execution
- Performance profiling and optimization hints
- Sensor simulation with GUI controls
- Advanced VM state inspection

### Phase 5: Polish & Examples (Planned)
- Example Smollu programs for hardware simulation
- Configuration save/load functionality
- Documentation and tutorials
- Export/sharing features

## Contributing

1. Follow Rust coding conventions
2. Add tests for new features
3. Update documentation
4. Ensure C integration remains safe

## License

MIT License - see parent project for details.

## Links

- [Parent Smollu Project](../)
- [Emulator Design Plan](../PLAN.md)
- [Language Specification](../doc/Language%20Spec.md)
- [Instruction Set](../doc/Instruction%20Set.md)