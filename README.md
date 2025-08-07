# Smollu - A small script language compiler and vm 

A lightweight custom script language compiler and VM implementation, written in pure C, that can run on embedded systems. 

This project aims to build a small script language for my own use and learning.

## Features

- [x] Support for basic arithmetic operations
- [x] Support for basic array operations
- [x] Support for basic function operations
- [x] Support for basic loop operations
- [x] Support for basic conditional operations
- [x] Support for system calls (native functions)

## Support platform

- [x] PC (Windows, Linux, MacOS)
- [ ] Zephyr RTOS

## Build

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Run Demo

First, compile the demo code:

```bash
cd build/demo/Simple\ demo
./smollu_compiler demo.smol -o demo.smolbc
```

Then, run the demo:
```bash
# Under build/demo/Simple\ demo
./smollu_demo
```

## Doc

- [x] Language Spec
- [x] Instruction Set
- [x] Bytecode format

## Project structure

```
smollu/
├── cmake/           # CMake files
├── demo/            # Demo code
    └── Simple demo/     # Simple demo code
├── src/             # Source code (C)
    ├── components/      # Components
        ├── compiler/    # Compiler
        └── vm/          # VM and interpreter
    └── smollu.h       # Public header file
├── test/            # Test code
└── doc/             # Documentation

```
