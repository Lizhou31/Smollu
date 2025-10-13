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

**Available LED Matrix Native Functions:**
- `led_matrix_init(rows, cols)` - Initialize matrix (1-64 x 1-64)
- `led_set_color(row, col, r, g, b)` - Set LED with RGB color (0-255)
- `led_set(row, col, state)` - Set LED on/off (0=off, 1=on)
- `led_clear()` - Clear all LEDs
- `led_set_row(row, pattern)` - Set row with bit pattern
- `led_set_col(col, pattern)` - Set column with bit pattern
- `led_get(row, col)` - Get LED state (returns 0/1)
- `delay_ms(milliseconds)` - Hardware-synchronized delay

Example usage:

```smol
init {
    native led_matrix_init(8, 8);
}

main {
    native led_set_color(0, 0, 255, 0, 0);  // Red LED at (0,0)
    native delay_ms(1000);                   // 1-second delay
    native led_clear();
}
```

## Compilers

Smollu has two compiler implementations targeting the same VM bytecode format:

### C Compiler (Production Ready)

The original C-based compiler provides stable, single-pass compilation.

**Pipeline:** `Source → Lexer → Parser → Bytecode Generator → VM Bytecode`

**Status:** ✅ Fully functional and production-ready

**Usage:**
```bash
./build/compiler/smollu_compiler input.smol -o output.smolbc
```

**Features:**
- Complete language support (arithmetic, control flow, functions, native calls)
- Automatic slot allocation for variables
- Comprehensive test coverage
- Direct bytecode emission (no intermediate representations)

### MLIR Compiler (Experimental)

Modern multi-stage compiler using LLVM's MLIR infrastructure for advanced optimizations.

**Current Status:** 🚧 Phase 1 Complete - High-level Smol dialect implemented

**Pipeline:**
```
Source → AST → Smol Dialect (high-level) → [Standard Dialects] → [SmolluASM] → Bytecode
```

**Working Now:**
```bash
# Generate AST
./build/compiler_mlir/smollu-mlir-compiler input.smol --emit-ast

# Generate high-level Smol MLIR
./build/compiler_mlir/smollu-mlir-compiler input.smol --emit-smol --target=rs-emulator

# Visualize AST
cd compiler_mlir/tools && python ast_visualizer.py demo.ast --format both
```

**Target System:**

Native calls are resolved at compile-time using YAML target definitions:
- `demo` - Basic target with `print` and `rand` (device_id: 0x00)
- `rs-emulator` - LED matrix support (device_id: 0x01, 9 native functions)

See `compiler_mlir/targets/README.md` for adding new targets.

**Roadmap:**

| Phase | Status | Description |
|-------|--------|-------------|
| 1. High-Level Dialect | ✅ Complete | Smol dialect with language semantics |
| 2. Low-Level Dialect | ⏸️ TODO | SmolluASM dialect (1:1 VM instructions) |
| 3. Lowering Passes | ⏸️ TODO | Multi-stage lowering with verifiers |
| 4. Generator Update | ⏸️ TODO | Verify pipeline compatibility |
| 5. Code Generation | ⏸️ TODO | Re-enable bytecode emission |
| 6. Build System | ⏸️ TODO | Integrate new dialects |
| 7. Compiler Driver | ⏸️ TODO | Full compilation pipeline |

**Next Steps:**
- Create SmolluASM dialect for VM instructions
- Implement lowering passes (Smol → Standard → SmolluASM)
- Re-enable bytecode generation

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

## Documentation

- Language Specification: `doc/Language Spec.md`
- VM Instruction Set: `doc/Instruction Set.md`
- Bytecode File Format: `doc/ByteCode format.md`

## Project Structure

```
smollu/
├── cmake/                          # Build system modules
│   ├── compiler.cmake
│   ├── compiler_mlir.cmake
│   ├── emulator.cmake
│   ├── tests.cmake
│   ├── vm.cmake
│   └── demo.cmake
│
├── vm/                             # Virtual Machine (C)
│   ├── smollu_vm.c
│   ├── smollu_vm.h
│   └── test/
│
├── compiler/                       # C Compiler (Production)
│   ├── smollu_lexer.c
│   ├── smollu_parser.c
│   ├── smollu_bytecode_codegen.c
│   ├── smollu_compiler.c
│   ├── smollu_native_tables.h
│   └── test/
│
├── compiler_mlir/                  # MLIR Compiler (Experimental)
│   ├── include/Smollu/             # Headers & TableGen definitions
│   │   ├── SmolDialect.{h,td}
│   │   ├── SmolOps.{h,td}
│   │   ├── Passes.h
│   │   └── ...
│   ├── lib/
│   │   ├── Lexer/
│   │   ├── Parser/
│   │   ├── Dialect/
│   │   ├── Pass/                   # Optimization passes
│   │   ├── Target/                 # Native function registry
│   │   └── CodeGen/                # ⏸️ Bytecode emitter (disabled)
│   ├── src/
│   │   └── smollu-mlir-compiler.cpp
│   ├── targets/                    # Target platform definitions (YAML)
│   └── tools/                      # AST visualization tools
│
├── emulator/                       # Rust Emulator + LED Matrix GUI
│   ├── src/
│   ├── examples/
│   └── c_integration/
│
├── demo/
│   ├── Simple demo/
│   └── LED matrix demo/
│
├── doc/
│   ├── Language Spec.md
│   ├── Instruction Set.md
│   └── ByteCode format.md
│
├── CLAUDE.md                       # Project instructions for Claude Code
├── CMakeLists.txt
└── CMakePresets.json
```
