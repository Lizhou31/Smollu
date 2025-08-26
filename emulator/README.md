# Smollu Emulator

A Rust-based emulator for the Smollu VM with GUI capabilities for hardware simulation. This emulator integrates the existing C VM core and provides additional debugging and simulation features.

## Overview

The Smollu Emulator is designed to provide a modern development environment for Smollu programs with:

- **Safe Rust Integration**: FFI bindings over the C VM with memory safety
- **Modern GUI Interface**: Real-time visualization with egui framework
- **Dual Mode Operation**: Both GUI and CLI interfaces available
- **Enhanced Debugging**: VM state inspection and execution control
- **LED Matrix Simulation**: Real-time hardware visualization with synchronized delays
- **Hardware Simulation**: Virtual GPIO pins and sensors (planned)

## Project Status

- ✅ **Phase 1 Complete**: Core C VM Integration
- ✅ **Phase 2 Complete**: Basic GUI Framework with file loading and console output
- ✅ **Phase 3 Complete**: LED Matrix Hardware Simulation with Synchronized Delays
- 📋 **Phase 4 Planned**: Additional Hardware Components (GPIO, sensors)

## Quick Start

### Prerequisites

- Rust 1.70+ with Cargo
- CMake 3.15+
- C compiler (GCC/Clang)
- The parent Smollu project built

### Building

The emulator is now integrated into the main project's unified build system. You can build everything with a single command:

**Unified Build (Recommended)**:
```bash
# From project root - builds C VM, compiler, and Rust emulator
cmake --preset=default
cmake --build build
```

**Manual Build** (if you need just the emulator):
```bash
# From emulator directory
cargo build
```

### Running

**GUI Mode (Default)**:
```bash
# Launch GUI without file
cargo run

# Launch GUI with file pre-loaded
cargo run "../build/demo/Simple demo/demo.smolbc"
```

**CLI Mode**:
```bash
# Run in console mode
cargo run -- --cli "../build/demo/Simple demo/demo.smolbc"
```

**LED Matrix Demos**:
```bash
# Run LED matrix animation demo
cargo run "../build/demo/LED matrix demo/animation_demo.smolbc"

# Run basic LED matrix demo
cargo run "../build/demo/LED matrix demo/basic_led_demo.smolbc"
```

**Examples and Tests**:
```bash
# Run integration tests
cargo run --example basic_test
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
│   ├── main.rs             # GUI/CLI launcher
│   ├── lib.rs              # Main library interface
│   ├── gui/                # GUI framework (Phase 2)
│   │   ├── mod.rs          # GUI module exports
│   │   ├── app.rs          # Main GUI application
│   │   └── widgets/        # GUI components
│   │       ├── mod.rs      # Widget exports
│   │       ├── console.rs  # Console output widget
│   │       ├── controls.rs # VM control buttons
│   │       └── led_matrix.rs # LED matrix visualization widget
│   ├── hardware/           # Hardware simulation (Phase 3)
│   │   ├── mod.rs          # Hardware module exports
│   │   └── led_matrix.rs   # LED matrix simulation with synchronized delays
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

#### 4. **GUI Framework** (`src/gui/`)
- **SmolluEmulatorApp**: Main GUI application with eframe integration
- **Console Widget**: Scrollable output display with auto-scroll
- **Controls Widget**: File loading, VM execution controls (Load/Run/Reset)
- **File Management**: Native file dialogs with .smolbc filtering

#### 5. **Hardware Simulation** (`src/hardware/`)
- **LED Matrix**: Configurable 2D matrix with RGB LED support
- **Synchronized Delays**: True hardware timing simulation with VM synchronization
- **Real-time GUI**: Live visualization with delay countdown display
- **Native Functions**: Complete API for matrix control (init, set, clear, patterns)

#### 6. **Dual Mode Interface** (`src/main.rs`)
- **GUI Mode**: Modern interface with real-time visualization
- **CLI Mode**: Traditional command-line interface with --cli flag
- **File Loading**: Support for both modes with identical functionality

## Usage Examples

### GUI Mode

The emulator launches in GUI mode by default:

```bash
# Launch empty GUI
cargo run

# Load file on startup
cargo run "../build/demo/Simple demo/demo.smolbc"
```

**GUI Features:**
- 📁 **File Loading**: Click "Load" button for native file picker
- ▶️ **VM Execution**: Run button starts threaded VM execution
- 🔄 **Reset**: Clear VM state and console output
- 📜 **Console**: Scrollable output with auto-scroll and clear options
- 🔲 **LED Matrix Panel**: Real-time hardware visualization with configuration controls
- ⚡ **Real-time**: Non-blocking execution with live status updates

### CLI Mode

For automated scripts or traditional workflows:

```bash
cargo run -- --cli "../build/demo/Simple demo/demo.smolbc"
```

### LED Matrix Hardware Simulation

The emulator includes comprehensive LED matrix hardware simulation with native functions for Smollu programs:

**Available Native Functions:**
```smol
native led_matrix_init(rows, cols);           // Initialize matrix (1-64 x 1-64)
native led_set_color(row, col, r, g, b);      // Set LED with RGB color (0-255)
native led_set(row, col, state);              // Set LED on/off (0=off, 1=on)
native led_clear();                           // Clear all LEDs
native led_set_row(row, pattern);             // Set row with bit pattern
native led_set_col(col, pattern);             // Set column with bit pattern
native led_get(row, col);                     // Get LED state (returns 0/1)
native delay_ms(milliseconds);                // Hardware-synchronized delay
```

**Example Smollu Program:**
```smol
init {
    native led_matrix_init(8, 8);
}

main {
    // Create a red dot that moves across the matrix
    local x = 0;
    while (x < 8) {
        native led_clear();
        native led_set_color(0, x, 255, 0, 0);  // Red LED
        native delay_ms(500);                   // Synchronized delay
        x = x + 1;
    }
}
```

**Hardware Features:**
- **RGB LEDs**: Full color support with 0-255 color values per channel
- **Configurable Size**: Up to 64x64 LED matrices
- **Pattern Operations**: Efficient row/column bit pattern setting
- **Synchronized Delays**: VM execution pauses with GUI countdown display
- **Real-time GUI**: Live LED state visualization with hover tooltips and statistics
- **Interactive Controls**: LED size, spacing, grid display, and color configuration

### Programmatic Usage

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
