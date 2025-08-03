# Smollu ByteCode Format

## Bytecode Format

1. Header (16 bytes)
    - magic number (4 bytes) - `SMOL`
    - version (1 byte)
    - device_id (1 byte)
    - function count (1 byte)
    - native function count (1 byte)
    - code_size (4 bytes)
    - reserved (4 bytes)

2. Native functions table (variable length, 2 bytes per entry)

3. Code (variable length)
    - Format:
        - Instruction (1 byte)
        - Operand (variable length)

## Example
```


```