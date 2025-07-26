# Smollu - A small script language interpreter and vm 

A lightweight custom script language interpreter and VM implementation, written in pure C, that can run on embedded systems. 

This project aims to build a small script language for my own use and learning.

## Features

- [ ] Support for basic arithmetic operations
- [ ] Support for basic array operations
- [ ] Support for basic function operations
- [ ] Support for basic loop operations
- [ ] Support for basic conditional operations
- [ ] Support for system calls

## Support platform

- [ ] PC (Windows, Linux, MacOS)
- [ ] Zephyr RTOS

## Doc

- [x] Language Spec
- [ ] Instruction Set

## Project structure

```
smollu/
├── src/            # Source code
    ├── components/      # Components
        ├── compiler/    # Compiler
        ├── vm/          # VM
        └── interpreter/ # Interpreter
    └── smollu.h       # Public header file
├── test/           # Test code
└── doc/            # Documentation

```
