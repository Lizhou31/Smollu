#ifndef DEMO_BYTECODE_H
#define DEMO_BYTECODE_H

#include <stdint.h>

/* ────────────────────────────────────────────────────────────────────────── */
/*  Byte-code opcode constants (duplicated from smollu_vm.c for testing)     */
/* ────────────────────────────────────────────────────────────────────────── */
enum {
    /* 00–0F Stack & const */
    OP_NOP = 0x00,
    OP_PUSH_NIL,
    OP_PUSH_TRUE,
    OP_PUSH_FALSE,
    OP_PUSH_I8,
    OP_PUSH_I32,
    OP_PUSH_F32,
    OP_DUP,
    OP_POP,
    OP_SWAP,

    /* 10–1F Locals / Globals */
    OP_LOAD_LOCAL = 0x10,
    OP_STORE_LOCAL,
    OP_LOAD_GLOBAL,
    OP_STORE_GLOBAL,

    /* 20–2F Arithmetic / logic */
    OP_ADD = 0x20,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_MOD,
    OP_NEG,
    OP_NOT,

    /* 30–3F Comparison */
    OP_EQ = 0x30,
    OP_NEQ,
    OP_LT,
    OP_LE,
    OP_GT,
    OP_GE,

    /* 40–4F Control flow */
    OP_JMP = 0x40,
    OP_JMP_IF_TRUE,
    OP_JMP_IF_FALSE,
    OP_HALT,

    /* 50–5F Function */
    OP_CALL = 0x50,
    OP_RET,

    /* 60–6F Native */
    OP_NCALL = 0x60,
    OP_SLEEP_MS,

    /* F0 Meta */
    OP_DBG_TRAP = 0xF0,
    OP_ILLEGAL = 0xFF,
};

/*
 * Demo bytecode: calculates 1 + 1, prints result using native #0, then halts.
 *
 */
static const uint8_t demo_bytecode[] = {
    OP_PUSH_NIL,
    OP_PUSH_NIL,
    OP_PUSH_I32, 1, 0, 0, 0,
    OP_PUSH_I32, 1, 0, 0, 0,
    OP_ADD,
    OP_STORE_LOCAL, 0,
    OP_PUSH_F32, 0x66, 0x66, 0x06, 0x40,
    OP_STORE_LOCAL, 1,
    OP_LOAD_LOCAL, 0,
    OP_LOAD_LOCAL, 1,
    OP_NCALL, 0, 2,
    OP_POP,
    OP_POP,      // end of init
    OP_PUSH_NIL, // local result
    OP_PUSH_I32, 0, 0, 0, 0,
    OP_STORE_GLOBAL, 0,
    OP_PUSH_F32, 0x66, 0x66, 0x06, 0x40,
    OP_STORE_GLOBAL, 1,
    OP_LOAD_GLOBAL, 0,
    OP_PUSH_I32, 10, 0, 0, 0,
    OP_LT,
    OP_JMP_IF_FALSE, 18, 0,
    OP_LOAD_GLOBAL, 0,
    OP_PUSH_I32, 1, 0, 0, 0,
    OP_ADD,
    OP_STORE_GLOBAL, 0,
    OP_LOAD_GLOBAL, 0,
    OP_NCALL, 0, 1,
    OP_JMP, 0xe3, 0xff,
    OP_LOAD_GLOBAL, 0,
    OP_PUSH_I32, 5, 0, 0, 0,
    OP_LT,
    OP_JMP_IF_FALSE, 10, 0,
    OP_LOAD_GLOBAL, 0,
    OP_PUSH_I32, 1, 0, 0, 0,
    OP_ADD,
    OP_STORE_GLOBAL, 0,
    OP_LOAD_GLOBAL, 0,
    OP_PUSH_I32, 8, 0, 0, 0,
    OP_LT,
    OP_JMP_IF_FALSE, 10, 0,
    OP_LOAD_GLOBAL, 0,
    OP_PUSH_I32, 2, 0, 0, 0,
    OP_ADD,
    OP_STORE_GLOBAL, 0,
    OP_LOAD_GLOBAL, 0,
    OP_PUSH_I32, 1, 0, 0, 0,
    OP_SUB,
    OP_STORE_GLOBAL, 0,
    OP_LOAD_GLOBAL, 0,
    OP_NCALL, 0, 1,
    OP_LOAD_GLOBAL, 0,
    OP_LOAD_GLOBAL, 1,
    OP_CALL, 140, 0x00, 2,  // temp
    OP_NCALL, 0, 1,
    OP_POP,  // end of main
    OP_HALT,
    OP_PUSH_NIL, // sum
    OP_LOAD_LOCAL, 0,
    OP_LOAD_LOCAL, 1,
    OP_ADD,
    OP_STORE_LOCAL, 2,
    OP_LOAD_LOCAL, 2,
    OP_STORE_GLOBAL, 0,
    OP_RET, 1,
};

static const size_t demo_bytecode_len = sizeof(demo_bytecode);

#endif /* DEMO_BYTECODE_H */ 