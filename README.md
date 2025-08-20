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
Run the compiled bytecode with enhanced debugging:

```bash
# From project root
cd emulator
cargo run -- "../build/demo/Simple demo/demo.smolbc"

# Or use the built executable
./emulator/target/debug/smollu-emulator "build/demo/Simple demo/demo.smolbc"
```

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
│   └── Simple demo/    # Basic demo with .smol source
├── doc/                # Language and system documentation
└── CMakePresets.json   # Build presets configuration
```
