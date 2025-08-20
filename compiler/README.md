# Smollu Compiler

The Smollu compiler is a three-stage compiler pipeline that transforms Smollu source code into bytecode for the Smollu VM.

## Architecture

The compiler consists of three main stages:

1. **Lexer** (`smollu_lexer.c`) - Tokenizes source code according to language specification
2. **Parser** (`smollu_parser.c`) - Builds Abstract Syntax Tree (AST) using recursive descent parsing
3. **Bytecode Generator** (`smollu_bytecode_codegen.c`) - Generates stack-based bytecode from AST

## Files

### Core Components
- `smollu_lexer.c` - Lexical analyzer
- `smollu_parser.c` - Syntax analyzer and AST builder  
- `smollu_bytecode_codegen.c` - Code generator
- `smollu_compiler.c` - Main compiler executable
- `smollu_compiler.h` - Compiler API definitions
- `smollu_native_tables.h` - Native function tables

### Tests
- `test/test_smollu_lexer.c` - Lexer unit tests
- `test/test_smollu_parser.c` - Parser unit tests
- `test/test_smollu_bytecode_codegen.c` - Bytecode generator unit tests

## Building

The compiler can be built as part of the main project or independently:

```bash
# Build as part of main project (recommended)
cd <project_root>
cmake --preset=default
cmake --build build

# Build compiler component independently
cd compiler/
mkdir build && cd build
cmake ..
cmake --build .
```

## Testing

Compiler tests use the Criterion testing framework:

```bash
# Build with tests enabled
cmake --preset=with-tests
cmake --build build

# Run all compiler tests
cd build
./test_smollu_lexer --verbose
./test_smollu_parser --verbose
./test_smollu_bytecode_codegen --verbose
```

## Usage

### Command Line
```bash
# Compile .smol source to .smolbc bytecode
./smollu_compiler input.smol -o output.smolbc
```


## Language Features

The compiler supports the complete Smollu language including:
- Basic arithmetic (int/float with automatic promotion)
- Boolean operations and comparisons
- Local and global variables
- Control flow (if/elif/else, while loops)
- User-defined functions with parameters and return values
- Native function calls for system integration
- Three-section program structure: `init{}`, `main{}`, `functions{}`

See the main project documentation for complete language specification.