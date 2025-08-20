# Smollu Virtual Machine

The Smollu VM is a lightweight stack-based virtual machine designed for executing Smollu bytecode on embedded systems.

## Features

- Stack-based architecture with 255-instruction set
- Support for int/float arithmetic with automatic type promotion
- Boolean operations and comparisons
- Local and global variable management
- Function calls with parameters and return values
- Native function calls for system integration

## Files

- `smollu_vm.h` - VM API and Value type definitions
- `smollu_vm.c` - VM implementation
- `test/test_smollu_vm.c` - VM unit tests

## Building

The VM can be built as part of the main project or independently:

```bash
# Build as part of main project (recommended)
cd <project_root>
cmake --preset=default
cmake --build build

# Build VM component independently
cd vm/
mkdir build && cd build
cmake ..
cmake --build .
```

## Testing

VM tests use the Criterion testing framework:

```bash
# Build with tests enabled
cmake --preset=with-tests
cmake --build build

# Run VM tests specifically
cd build
./test_smollu_vm --verbose
```

## API Usage
TODO