# Smollu - A small script language compiler and vm

A lightweight custom script language compiler and VM implementation, written in pure C, that can run on embedded systems.

This project aims to build a small script language for my own use and learning.

## Features

- [x] Support for basic arithmetic operations
- [x] Support for basic function operations
- [x] Support for basic loop operations
- [x] Support for basic conditional operations
- [x] Support for system calls (native functions)
- [x] Rust emulator with VM state inspection and output capture
- [x] LED matrix hardware simulation with real-time GUI visualization

## Support platform

- [x] PC (Windows, Linux, MacOS)
- [ ] Zephyr RTOS

## Build

### Quick Start with CMake Presets

```bash
# Default build (C + Rust, no tests)
cmake --preset=default
cmake --build build

# With tests enabled
cmake --preset=with-tests
cmake --build build
ctest --test-dir build --verbose

# Release build
cmake --preset=release
cmake --build build

# Debug build
cmake --preset=debug
cmake --build build
```

### Manual Build

```bash
mkdir build && cd build
cmake .. -DBUILD_TESTS=ON
cmake --build .
```

The unified build system automatically builds:
- C VM (`smollu_vm`)
- C compiler (`smollu_compiler`)
- Rust emulator (`smollu-emulator`)
- All test suites (7 total: 4 C + 3 Rust)

## Run Demo

### Using C VM
First, compile the demo code:

```bash
cd build/demo/Simple\ demo
./smollu_compiler demo.smol -o demo.smolbc
```

Then, run with the C VM:
```bash
# Under build/demo/Simple\ demo
./smollu_demo
```

### Using Rust Emulator

Run the compiled bytecode with enhanced debugging and GUI:

```bash
# From project root
cd emulator
cargo run -- "../build/demo/Simple demo/demo.smolbc"

# Or use the built executable
./emulator/target/debug/smollu-emulator "build/demo/Simple demo/demo.smolbc"

# CLI mode (no GUI)
cargo run -- --cli "../build/demo/Simple demo/demo.smolbc"
```

### LED Matrix Hardware Simulation

The emulator includes interactive LED matrix simulation with hardware-accurate timing:

```bash
# From project root
cd emulator

# Run LED matrix animation demo (moving dot with colors and delays)
cargo run "../build/demo/LED matrix demo/animation_demo.smolbc"

# Run basic LED matrix patterns demo
cargo run "../build/demo/LED matrix demo/basic_led_demo.smolbc"
```

**Hardware Simulation Features:**
- **Real-time LED Matrix**: Interactive 2D LED display with RGB color support
- **Synchronized Delays**: VM execution pauses during `delay_ms()` with GUI countdown
- **Native Functions**: 9 functions for complete LED matrix control
- **GUI Controls**: Configurable LED size, spacing, colors, and grid display
- **Hardware Timing**: Accurate delay simulation for embedded system development

## Testing

The project includes comprehensive test coverage with 7 test suites:

**C Tests (Criterion framework):**
- `vm_tests` - VM execution and bytecode interpretation
- `lexer_tests` - Tokenization and lexical analysis
- `parser_tests` - AST generation and syntax parsing
- `bytecode_codegen_tests` - Bytecode generation from AST

**Rust Tests:**
- `emulator_tests` - Unit tests for Rust components
- `emulator_basic_test` - Comprehensive integration test
- `emulator_simple_test` - Basic functionality validation

Run all tests:
```bash
ctest --test-dir build --verbose
```

## LED Matrix Native Functions

The emulator provides comprehensive LED matrix control through native functions:

```smol
init {
    native led_matrix_init(8, 8);          // Initialize 8x8 LED matrix
}

main {
    // Set individual LED colors (row, col, red, green, blue)
    native led_set_color(0, 0, 255, 0, 0);  // Red LED at (0,0)
    native led_set_color(0, 1, 0, 255, 0);  // Green LED at (0,1)
    native led_set_color(0, 2, 0, 0, 255);  // Blue LED at (0,2)

    native delay_ms(1000);                  // Synchronized 1-second delay

    native led_clear();                     // Clear all LEDs

    // Set row/column patterns with bit masks
    native led_set_row(0, 0b10101010);      // Alternating pattern on row 0
    native led_set_col(0, 0b11110000);      // Top half pattern on col 0

    // Query LED state
    local state = native led_get(0, 0);     // Returns 0 (off) or 1 (on)
}
```

**Available Functions:**
- `led_matrix_init(rows, cols)` - Initialize matrix (1-64 x 1-64)
- `led_set_color(row, col, r, g, b)` - Set LED with RGB color (0-255)
- `led_set(row, col, state)` - Set LED on/off (0=off, 1=on)
- `led_clear()` - Clear all LEDs
- `led_set_row(row, pattern)` - Set row with bit pattern
- `led_set_col(col, pattern)` - Set column with bit pattern
- `led_get(row, col)` - Get LED state (returns 0/1)
- `delay_ms(milliseconds)` - Hardware-synchronized delay

## Documentation

- [x] Language Spec (`doc/Language Spec.md`)
- [x] Instruction Set (`doc/Instruction Set.md`)
- [x] Bytecode format (`doc/ByteCode format.md`)

## Project Structure

```
smollu/
├── cmake/              # CMake modules and configuration
│   ├── emulator.cmake  # Rust integration module
│   ├── tests.cmake     # Unified test registration
│   └── ...            # Component-specific modules
├── vm/                 # Virtual Machine (C)
│   ├── smollu_vm.c     # VM implementation
│   ├── smollu_vm.h     # VM public API
│   └── test/           # VM test suite
├── compiler/           # Compiler toolchain (C)
│   ├── smollu_lexer.c      # Lexical analyzer
│   ├── smollu_parser.c     # Syntax parser
│   ├── smollu_bytecode_codegen.c  # Code generator
│   ├── smollu_compiler.c   # Main compiler executable
│   └── test/               # Compiler test suites
├── emulator/           # Rust emulator with FFI integration
│   ├── src/            # Rust source code
│   ├── examples/       # Integration tests
│   └── c_integration/  # C wrapper API
├── demo/               # Example programs
│   ├── Simple demo/    # Basic demo with .smol source
│   └── LED matrix demo/  # LED matrix hardware simulation demos
├── doc/                # Language and system documentation
└── CMakePresets.json   # Build presets configuration
```
