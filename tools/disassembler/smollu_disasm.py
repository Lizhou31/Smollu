#!/usr/bin/env python3
"""
Smollu Bytecode Disassembler

Converts .smolbc bytecode files to readable assembly format with:
- Hex offsets and raw opcodes
- Instruction mnemonics and operands
- Jump labels for control flow
- Section headers

Usage:
    python smollu_disasm.py input.smolbc [output.asm]
"""

import sys
import struct
from typing import BinaryIO, Dict, List, Tuple, Optional


class SmolluDisassembler:
    """Disassembles Smollu bytecode (.smolbc) files to assembly format."""

    # Instruction set definitions
    INSTRUCTIONS = {
        # Stack & constants (0x00 - 0x0F)
        0x00: ("NOP", 0, ""),
        0x01: ("PUSH_NIL", 0, ""),
        0x02: ("PUSH_TRUE", 0, ""),
        0x03: ("PUSH_FALSE", 0, ""),
        0x04: ("PUSH_I8", 1, "b"),      # signed byte
        0x05: ("PUSH_I32", 4, "<i"),    # signed 32-bit int
        0x06: ("PUSH_F32", 4, "<f"),    # 32-bit float
        0x07: ("DUP", 0, ""),
        0x08: ("POP", 0, ""),
        0x09: ("SWAP", 0, ""),

        # Local/global vars (0x10 - 0x1F)
        0x10: ("LOAD_LOCAL", 1, "B"),   # unsigned byte
        0x11: ("STORE_LOCAL", 1, "B"),
        0x12: ("LOAD_GLOBAL", 1, "B"),
        0x13: ("STORE_GLOBAL", 1, "B"),

        # Arithmetic/logic (0x20 - 0x2F)
        0x20: ("ADD", 0, ""),
        0x21: ("SUB", 0, ""),
        0x22: ("MUL", 0, ""),
        0x23: ("DIV", 0, ""),
        0x24: ("MOD", 0, ""),
        0x25: ("NEG", 0, ""),
        0x26: ("NOT", 0, ""),
        0x27: ("AND", 0, ""),
        0x28: ("OR", 0, ""),

        # Comparison (0x30 - 0x3F)
        0x30: ("EQ", 0, ""),
        0x31: ("NEQ", 0, ""),
        0x32: ("LT", 0, ""),
        0x33: ("LE", 0, ""),
        0x34: ("GT", 0, ""),
        0x35: ("GE", 0, ""),

        # Control flow (0x40 - 0x4F)
        0x40: ("JMP", 2, "<h"),         # signed 16-bit offset
        0x41: ("JMP_IF_TRUE", 2, "<h"),
        0x42: ("JMP_IF_FALSE", 2, "<h"),
        0x43: ("HALT", 0, ""),

        # Function calls (0x50 - 0x5F)
        0x50: ("CALL", 2, "BB"),        # func_id, argc
        0x51: ("RET", 1, "B"),          # return value count

        # Native interface (0x60 - 0x6F)
        0x60: ("NCALL", 2, "BB"),       # native_id, argc
        0x61: ("SLEEP_MS", 2, "<H"),    # unsigned 16-bit

        # Debug/meta (0xF0 - 0xFF)
        0xF0: ("DBG_TRAP", 0, ""),
        0xFF: ("ILLEGAL", 0, ""),
    }

    def __init__(self, bytecode: bytes):
        self.bytecode = bytecode
        self.pos = 0
        self.code_start = 0
        self.jump_targets = set()

        # Header fields
        self.magic = ""
        self.version = 0
        self.device_id = 0
        self.func_count = 0
        self.native_count = 0
        self.code_size = 0
        self.reserved = 0

        # Native function table
        self.natives = []

    def read_bytes(self, n: int) -> bytes:
        """Read n bytes from current position."""
        data = self.bytecode[self.pos:self.pos + n]
        self.pos += n
        return data

    def read_u8(self) -> int:
        """Read unsigned 8-bit integer."""
        return struct.unpack("B", self.read_bytes(1))[0]

    def read_u16(self) -> int:
        """Read unsigned 16-bit integer (little-endian)."""
        return struct.unpack("<H", self.read_bytes(2))[0]

    def read_u32(self) -> int:
        """Read unsigned 32-bit integer (little-endian)."""
        return struct.unpack("<I", self.read_bytes(4))[0]

    def parse_header(self) -> None:
        """Parse 16-byte bytecode header."""
        self.magic = self.read_bytes(4).decode('ascii', errors='replace')
        self.version = self.read_u8()
        self.device_id = self.read_u8()
        self.func_count = self.read_u8()
        self.native_count = self.read_u8()
        self.code_size = self.read_u32()
        self.reserved = self.read_u32()

    def parse_native_table(self) -> None:
        """Parse native function table (2 bytes per entry)."""
        for i in range(self.native_count):
            native_id = self.read_u16()
            self.natives.append(native_id)

        self.code_start = self.pos

    def find_jump_targets(self) -> None:
        """First pass: identify all jump targets for label generation."""
        saved_pos = self.pos
        self.pos = self.code_start

        while self.pos < len(self.bytecode):
            opcode = self.read_u8()

            if opcode not in self.INSTRUCTIONS:
                # Unknown opcode, skip
                continue

            mnemonic, operand_size, fmt = self.INSTRUCTIONS[opcode]

            # Check if this is a jump instruction
            if mnemonic in ("JMP", "JMP_IF_TRUE", "JMP_IF_FALSE"):
                offset_bytes = self.read_bytes(operand_size)
                offset = struct.unpack(fmt, offset_bytes)[0]
                target = self.pos + offset
                self.jump_targets.add(target)
            else:
                # Skip operand bytes
                self.read_bytes(operand_size)

        self.pos = saved_pos

    def disassemble_instruction(self, offset: int) -> Tuple[str, int]:
        """
        Disassemble one instruction at current position.

        Returns:
            (assembly_line, bytes_consumed)
        """
        start_pos = self.pos
        opcode = self.read_u8()

        if opcode not in self.INSTRUCTIONS:
            # Unknown opcode
            hex_bytes = f"{opcode:02x}"
            return f"{offset:04x}: {hex_bytes:20s}  UNKNOWN 0x{opcode:02x}", 1

        mnemonic, operand_size, fmt = self.INSTRUCTIONS[opcode]

        # Build hex byte string
        hex_bytes = f"{opcode:02x}"
        operand_str = ""

        if operand_size > 0:
            operand_bytes = self.read_bytes(operand_size)
            hex_bytes += " " + " ".join(f"{b:02x}" for b in operand_bytes)

            # Decode operand based on instruction type
            if mnemonic == "PUSH_I8":
                val = struct.unpack(fmt, operand_bytes)[0]
                operand_str = f"{val}"

            elif mnemonic == "PUSH_I32":
                val = struct.unpack(fmt, operand_bytes)[0]
                operand_str = f"{val}"

            elif mnemonic == "PUSH_F32":
                val = struct.unpack(fmt, operand_bytes)[0]
                operand_str = f"{val}"

            elif mnemonic in ("LOAD_LOCAL", "STORE_LOCAL", "LOAD_GLOBAL", "STORE_GLOBAL"):
                slot = struct.unpack(fmt, operand_bytes)[0]
                operand_str = f"{slot}"

            elif mnemonic in ("JMP", "JMP_IF_TRUE", "JMP_IF_FALSE"):
                offset_val = struct.unpack(fmt, operand_bytes)[0]
                target = self.pos + offset_val
                operand_str = f"label_{target:04x}  ; offset={offset_val:+d}"

            elif mnemonic == "CALL":
                func_id = operand_bytes[0]
                argc = operand_bytes[1]
                operand_str = f"func_{func_id}, {argc}"

            elif mnemonic == "RET":
                rv_count = operand_bytes[0]
                operand_str = f"{rv_count}"

            elif mnemonic == "NCALL":
                native_id = operand_bytes[0]
                argc = operand_bytes[1]
                operand_str = f"native_{native_id}, {argc}"

            elif mnemonic == "SLEEP_MS":
                ms = struct.unpack(fmt, operand_bytes)[0]
                operand_str = f"{ms}"

            else:
                # Generic hex display for other operands
                operand_str = " ".join(f"0x{b:02x}" for b in operand_bytes)

        # Format the line
        instruction = f"{mnemonic:16s} {operand_str}" if operand_str else mnemonic
        line = f"{offset:04x}: {hex_bytes:20s}  {instruction}"

        bytes_consumed = self.pos - start_pos
        return line, bytes_consumed

    def disassemble(self) -> str:
        """Disassemble the entire bytecode file."""
        output = []

        # Parse header
        self.parse_header()

        output.append("; ===== HEADER =====")
        output.append(f"; Magic: {self.magic}")
        output.append(f"; Version: {self.version}, Device ID: {self.device_id}")
        output.append(f"; Function count: {self.func_count}, Native functions: {self.native_count}")
        output.append(f"; Code size: {self.code_size} bytes")
        output.append(f"; Reserved: 0x{self.reserved:08x}")
        output.append("")

        # Parse native function table
        self.parse_native_table()

        if self.natives:
            output.append("; ===== NATIVE FUNCTIONS =====")
            for i, native_id in enumerate(self.natives):
                output.append(f"; Native #{i}: 0x{native_id:04x}")
            output.append("")

        # Find all jump targets
        self.find_jump_targets()

        # Disassemble code section
        output.append("; ===== CODE =====")
        self.pos = self.code_start

        while self.pos < len(self.bytecode):
            offset = self.pos - self.code_start

            # Add label if this is a jump target
            if self.pos in self.jump_targets:
                output.append(f"label_{self.pos:04x}:")

            line, consumed = self.disassemble_instruction(offset)
            output.append(line)

        return "\n".join(output)


def main():
    """Main entry point for the disassembler."""
    if len(sys.argv) < 2:
        print("Usage: python smollu_disasm.py input.smolbc [output.asm]")
        print("\nDisassembles Smollu bytecode files to readable assembly format.")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    # Read bytecode file
    try:
        with open(input_file, 'rb') as f:
            bytecode = f.read()
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found")
        sys.exit(1)
    except IOError as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Disassemble
    disasm = SmolluDisassembler(bytecode)
    asm_output = disasm.disassemble()

    # Write output
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(asm_output)
            print(f"Disassembled to: {output_file}")
        except IOError as e:
            print(f"Error writing output file: {e}")
            sys.exit(1)
    else:
        print(asm_output)


if __name__ == "__main__":
    main()
