# Smollu Disassembler

A Python-based disassembler for Smollu bytecode (`.smolbc`) files that converts binary bytecode to readable assembly format.

## Features

- **Complete Instruction Set Support**: Disassembles all 50+ Smollu VM instructions
- **Detailed Output**: Shows hex offsets, raw opcodes, mnemonics, and operands
- **Jump Labels**: Automatically generates labels for control flow targets
- **Section Headers**: Clearly marks HEADER, NATIVE FUNCTIONS, and CODE sections
- **Standalone**: Pure Python 3, no external dependencies required

## Installation

No installation required! Just Python 3.6+ is needed.

```bash
# Make the script executable (optional)
chmod +x smollu_disasm.py
```

## Usage

### Basic Usage

```bash
# Print disassembly to stdout
python smollu_disasm.py input.smolbc

# Save disassembly to file
python smollu_disasm.py input.smolbc output.asm
```

### Examples

```bash
# Disassemble the demo program
python smollu_disasm.py ../../build/demo/Simple\ demo/demo.smolbc

# Save output for analysis
python smollu_disasm.py demo.smolbc demo.asm

# Use from project root
python tools/disassembler/smollu_disasm.py build/demo/Simple\ demo/demo.smolbc
```

## Output Format

The disassembler produces assembly output with the following structure:

```
; ===== HEADER =====
; Magic: SMOL
; Version: 0, Device ID: 0
; Function count: 3, Native functions: 1
; Code size: 197 bytes
; Reserved: 0x00000000

; ===== NATIVE FUNCTIONS =====
; Native #0: 0x0000
; Native #1: 0x0100

; ===== CODE =====
0x0000: 40 10 00          JMP              label_0012  ; offset=+16
0x0003: 10 00             LOAD_LOCAL       0
0x0005: 10 01             LOAD_LOCAL       1
0x0007: 20                ADD
0x0008: 11 02             STORE_LOCAL      2
0x000a: 10 02             LOAD_GLOBAL      2
0x000c: 13 00             STORE_GLOBAL     0
label_0012:
0x0012: 10 02             LOAD_LOCAL       2
0x0014: 51 01             RET              1
```

### Output Components

1. **Header Section**: Bytecode metadata
   - Magic number (should be "SMOL")
   - Version and device ID
   - Function and native function counts
   - Code size

2. **Native Functions Section**: Lists registered native functions
   - Native function IDs (as defined in bytecode)
   - Indexed from 0

3. **Code Section**: Disassembled instructions
   - **Offset** (hex): Position from code section start
   - **Raw bytes** (hex): Opcode and operand bytes
   - **Mnemonic**: Instruction name
   - **Operands**: Decoded operands with labels for jumps

## Instruction Set Reference

### Stack & Constants (0x00 - 0x0F)

| Opcode | Mnemonic      | Operands  | Description                    |
|--------|---------------|-----------|--------------------------------|
| 0x00   | NOP           | -         | No operation                   |
| 0x01   | PUSH_NIL      | -         | Push nil value                 |
| 0x02   | PUSH_TRUE     | -         | Push boolean true              |
| 0x03   | PUSH_FALSE    | -         | Push boolean false             |
| 0x04   | PUSH_I8       | i8        | Push 8-bit signed integer      |
| 0x05   | PUSH_I32      | i32       | Push 32-bit signed integer     |
| 0x06   | PUSH_F32      | f32       | Push 32-bit float              |
| 0x07   | DUP           | -         | Duplicate top of stack         |
| 0x08   | POP           | -         | Pop top of stack               |
| 0x09   | SWAP          | -         | Swap top two stack values      |

### Variables (0x10 - 0x1F)

| Opcode | Mnemonic      | Operands  | Description                    |
|--------|---------------|-----------|--------------------------------|
| 0x10   | LOAD_LOCAL    | u8        | Load local variable (slot 0-255)|
| 0x11   | STORE_LOCAL   | u8        | Store to local variable        |
| 0x12   | LOAD_GLOBAL   | u8        | Load global variable           |
| 0x13   | STORE_GLOBAL  | u8        | Store to global variable       |

### Arithmetic & Logic (0x20 - 0x2F)

| Opcode | Mnemonic | Description              |
|--------|----------|--------------------------|
| 0x20   | ADD      | Addition                 |
| 0x21   | SUB      | Subtraction              |
| 0x22   | MUL      | Multiplication           |
| 0x23   | DIV      | Division (int truncates) |
| 0x24   | MOD      | Modulo                   |
| 0x25   | NEG      | Unary negation           |
| 0x26   | NOT      | Logical NOT              |
| 0x27   | AND      | Logical AND              |
| 0x28   | OR       | Logical OR               |

### Comparison (0x30 - 0x3F)

| Opcode | Mnemonic | Description           |
|--------|----------|-----------------------|
| 0x30   | EQ       | Equal                 |
| 0x31   | NEQ      | Not equal             |
| 0x32   | LT       | Less than             |
| 0x33   | LE       | Less or equal         |
| 0x34   | GT       | Greater than          |
| 0x35   | GE       | Greater or equal      |

### Control Flow (0x40 - 0x4F)

| Opcode | Mnemonic      | Operands | Description                     |
|--------|---------------|----------|---------------------------------|
| 0x40   | JMP           | i16      | Unconditional jump (relative)   |
| 0x41   | JMP_IF_TRUE   | i16      | Jump if top is true             |
| 0x42   | JMP_IF_FALSE  | i16      | Jump if top is false            |
| 0x43   | HALT          | -        | Stop execution                  |

### Functions (0x50 - 0x5F)

| Opcode | Mnemonic | Operands     | Description                    |
|--------|----------|--------------|--------------------------------|
| 0x50   | CALL     | func_id, argc| Call script function           |
| 0x51   | RET      | rv_count     | Return from function           |

### Native Interface (0x60 - 0x6F)

| Opcode | Mnemonic  | Operands      | Description                    |
|--------|-----------|---------------|--------------------------------|
| 0x60   | NCALL     | native_id, argc| Call native C function        |
| 0x61   | SLEEP_MS  | u16           | Sleep for N milliseconds       |

### Debug & Meta (0xF0 - 0xFF)

| Opcode | Mnemonic  | Description               |
|--------|-----------|---------------------------|
| 0xF0   | DBG_TRAP  | Debug breakpoint          |
| 0xFF   | ILLEGAL   | Illegal instruction       |

## Bytecode Format

The `.smolbc` file format consists of:

### Header (16 bytes)
```
Offset  Size  Field
------  ----  -----
0x00    4     Magic number: "SMOL" (0x53 0x4D 0x4F 0x4C)
0x04    1     Version
0x05    1     Device ID
0x06    1     Function count
0x07    1     Native function count
0x08    4     Code size (little-endian)
0x0C    4     Reserved
```

### Native Function Table (variable length)
- 2 bytes per native function (function ID as u16, little-endian)
- Total length: `native_count * 2` bytes

### Code Section (variable length)
- Bytecode instructions with variable-length encoding
- Length specified in header's `code_size` field

## Troubleshooting

### Invalid Magic Number
If you see a warning about magic number not being "SMOL", the file may be:
- Not a valid `.smolbc` file
- Corrupted during transfer
- From an incompatible compiler version

### Unknown Opcodes
The disassembler will show `UNKNOWN 0xXX` for unrecognized opcodes:
- May indicate bytecode from a newer VM version
- Could be corrupted data
- Check that the file is valid bytecode

### Missing Jump Labels
If jump targets seem incorrect:
- Verify the bytecode file is complete
- Check that relative offsets aren't corrupted
- Ensure the code section starts at the correct position

## Related Documentation

- **Language Specification**: `../../doc/Language Spec.md`
- **Instruction Set**: `../../doc/Instruction Set.md`
- **Bytecode Format**: `../../doc/ByteCode format.md`
- **VM Implementation**: `../../vm/smollu_vm.c`
- **Compiler**: `../../compiler/smollu_compiler.c`

## Examples

### Example 1: Simple Arithmetic

Source code:
```smol
init {
  local x = 10;
  local y = 20;
  local sum = x + y;
}
```

Disassembly:
```
0x0000: 05 0a 00 00 00    PUSH_I32         10
0x0005: 11 00             STORE_LOCAL      0
0x0007: 05 14 00 00 00    PUSH_I32         20
0x000c: 11 01             STORE_LOCAL      1
0x000e: 10 00             LOAD_LOCAL       0
0x0010: 10 01             LOAD_LOCAL       1
0x0012: 20                ADD
0x0013: 11 02             STORE_LOCAL      2
```

### Example 2: Control Flow

Source code:
```smol
init {
  local x = 5;
  if (x > 3) {
    x = 10;
  }
}
```

Disassembly:
```
0x0000: 05 05 00 00 00    PUSH_I32         5
0x0005: 11 00             STORE_LOCAL      0
0x0007: 10 00             LOAD_LOCAL       0
0x0009: 05 03 00 00 00    PUSH_I32         3
0x000e: 34                GT
0x000f: 42 07 00          JMP_IF_FALSE     label_0019  ; offset=+7
0x0012: 05 0a 00 00 00    PUSH_I32         10
0x0017: 11 00             STORE_LOCAL      0
label_0019:
0x0019: 43                HALT
```

## License

Part of the Smollu project. See project root for license information.

## Contributing

This tool is part of the Smollu compiler toolchain. For issues or improvements:
1. Check existing bytecode format documentation
2. Verify against the VM implementation
3. Test with various `.smolbc` files
4. Submit improvements via the main project repository
