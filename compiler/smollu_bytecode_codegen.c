/**
 * @file smollu_bytecode_codegen.c
 * @author Lizhou (lisie31s@gmail.com)
 * @brief Bytecode generator translating Smollu AST into executable byte stream
 * @version 0.1
 * @date 2025-08-07
 * 
 * @copyright Copyright (c) 2025
 *
 *  The codegen is intentionally kept independent from the parser – it only
 *  consumes the public AST described in `smollu_compiler.h`.  Callers are
 *  responsible for feeding it a fully-built AST tree.
 *
 *  ────────────────────────────────────────────────────────────────────────────
 *   Header (16 bytes)
 *  ────────────────────────────────────────────────────────────────────────────
 *   0  – 3   "SMOL"          magic
 *   4        version         (caller supplied)
 *   5        device_id       (caller supplied)
 *   6        function_count  (filled in during generation)
 *   7        native_count    (filled in during generation)
 *   8  – 11  code_size       (filled in – size of code segment ONLY)
 *  12  – 15  reserved        (0 for now)
 *
 *  Immediately after the header comes the native function table
 *  (2 bytes/index).  The order of entries is deterministic – first appearance
 *  of a native call assigns the next free slot.  The table is used by the VM
 *  to translate a one-byte native id embedded in OP_NCALL into a host
 *  function pointer.
 */

#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "smollu_compiler.h" /* Public AST / token definitions */
#include "smollu_native_tables.h"  /* device native mapping */

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Opcode enumeration (duplicated from smollu_vm.c to avoid cyclic include)   */
/* ──────────────────────────────────────────────────────────────────────────── */

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
    OP_AND,
    OP_OR,

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

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Utility: simple automatically-growing byte buffer                          */
/* ──────────────────────────────────────────────────────────────────────────── */

typedef struct {
    uint8_t *data;
    size_t   len;
    size_t   cap;
} ByteBuf;

static void bb_init(ByteBuf *bb) {
    bb->data = NULL;
    bb->len  = 0;
    bb->cap  = 0;
}

static void bb_reserve(ByteBuf *bb, size_t extra) {
    if (bb->len + extra <= bb->cap) return;
    size_t new_cap = bb->cap ? bb->cap * 2 : 128;
    while (new_cap < bb->len + extra) new_cap *= 2;
    uint8_t *tmp = (uint8_t *)realloc(bb->data, new_cap);
    if (!tmp) {
        fprintf(stderr, "[Codegen] Failed to allocate %zu bytes (buf grow)\n", new_cap);
        exit(1);
    }
    bb->data = tmp;
    bb->cap  = new_cap;
}

static size_t bb_write_u8 (ByteBuf *bb, uint8_t v)  { bb_reserve(bb, 1); bb->data[bb->len++] = v; return bb->len - 1; }
static size_t bb_write_u16(ByteBuf *bb, uint16_t v) { bb_reserve(bb, 2); bb->data[bb->len++] = v & 0xFF; bb->data[bb->len++] = v >> 8; return bb->len - 2; }
static size_t bb_write_u32(ByteBuf *bb, uint32_t v) { bb_reserve(bb, 4);
                                                      for (int i = 0; i < 4; i++) bb->data[bb->len++] = (v >> (8*i)) & 0xFF; return bb->len - 4; }
static void   bb_patch_u16(ByteBuf *bb, size_t at, uint16_t v) { assert(at + 1 < bb->len); bb->data[at] = v & 0xFF; bb->data[at+1] = v >> 8; }
static void   bb_patch_u32(ByteBuf *bb, size_t at, uint32_t v); /* fwd */

static void bb_free(ByteBuf *bb) { free(bb->data); }

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Very small string → int map (open-addressing, power-of-two)                */
/*  Enough for dozens of identifiers, no removal needed.                      */
/* ──────────────────────────────────────────────────────────────────────────── */

#define MAP_LOAD_FACTOR 0.75f

typedef struct {
    char   *key;
    int     value;
} MapEntry;

typedef struct {
    MapEntry *entries;
    size_t    cap;
    size_t    count;
} StrIntMap;

static uint32_t hash_str(const char *s) {
    /* FNV-1a 32-bit */
    uint32_t h = 2166136261u;
    for (; *s; s++) {
        h ^= (uint8_t)(*s);
        h *= 16777619u;
    }
    return h;
}

static void map_init(StrIntMap *m) {
    m->entries = NULL;
    m->cap = m->count = 0;
}

static void map_grow(StrIntMap *m) {
    size_t new_cap = m->cap ? m->cap * 2 : 16;
    MapEntry *new_entries = calloc(new_cap, sizeof(MapEntry));
    if (!new_entries) {
        fprintf(stderr, "[Codegen] map_grow OOM\n");
        exit(1);
    }
    /* rehash */
    for (size_t i = 0; i < m->cap; i++) {
        if (!m->entries[i].key) continue;
        uint32_t idx = hash_str(m->entries[i].key) & (new_cap - 1);
        while (new_entries[idx].key) idx = (idx + 1) & (new_cap - 1);
        new_entries[idx] = m->entries[i];
    }
    free(m->entries);
    m->entries = new_entries;
    m->cap = new_cap;
}

static int map_lookup(StrIntMap *m, const char *key) {
    if (!m->cap) return -1;
    uint32_t idx = hash_str(key) & (m->cap - 1);
    for (;;) {
        MapEntry *e = &m->entries[idx];
        if (!e->key) return -1;
        if (strcmp(e->key, key) == 0) return e->value;
        idx = (idx + 1) & (m->cap - 1);
    }
}

static int map_insert(StrIntMap *m, const char *key, int value) {
    if ((float)(m->count + 1) / (float)(m->cap ? m->cap : 1) > MAP_LOAD_FACTOR) {
        map_grow(m);
    }
    if (!m->cap) map_grow(m); /* ensure capacity */
    uint32_t idx = hash_str(key) & (m->cap - 1);
    while (m->entries[idx].key) idx = (idx + 1) & (m->cap - 1);
    m->entries[idx].key = strdup(key);
    m->entries[idx].value = value;
    m->count++;
    return value;
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Forward declarations                                                       */
/* ──────────────────────────────────────────────────────────────────────────── */

typedef struct Compiler Compiler;

static void compile_block       (Compiler *c, ASTNode *block, bool is_func_body);
static void compile_expression  (Compiler *c, ASTNode *expr);

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Compiler-wide context                                                      */
/* ──────────────────────────────────────────────────────────────────────────── */

struct FunctionInfo {
    const char *name;  /* borrowed from AST – keep until end */
    uint16_t    address; /* filled once emitted */
    uint8_t     arity;   /* number of parameters */
};

struct PendingCallPatch {
    size_t   addr_offset;  /* location in buffer where 16-bit addr is to patch */
    const char *target_name; /* function name */
    struct PendingCallPatch *next;
};

struct Compiler {
    ByteBuf  code;

    /* symbol tables */
    StrIntMap globals; /* name -> slot (uint8) */
    const DeviceNativeTable *device_tbl;
    size_t   code_base;

    /* function registry (simple array) */
    struct FunctionInfo *functions;
    size_t   func_count;
    size_t   func_cap;

    /* patches */
    struct PendingCallPatch *patches;

    /* current function state */
    StrIntMap locals; /* reset per function */
    uint8_t   local_count;
};

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Error helper                                                               */
/* ──────────────────────────────────────────────────────────────────────────── */

#define CODEGEN_ERROR(fmt, ...) do { \
    fprintf(stderr, "[Codegen] " fmt "\n", ##__VA_ARGS__); \
    exit(1); } while (0)

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Patching helpers                                                           */
/* ──────────────────────────────────────────────────────────────────────────── */

static void register_patch(Compiler *c, size_t addr_offset, const char *target_name) {
    struct PendingCallPatch *p = malloc(sizeof(*p));
    if (!p) {
        fprintf(stderr, "[Codegen] OOM patch alloc\n");
        exit(1);
    }
    p->addr_offset = addr_offset;
    p->target_name = target_name;
    p->next = c->patches;
    c->patches = p;
}

static void resolve_patches(Compiler *c) {
    for (struct PendingCallPatch *p = c->patches; p; p = p->next) {
        /* look up function address */
        uint16_t addr = 0xFFFF;
        for (size_t i = 0; i < c->func_count; i++) {
            if (strcmp(c->functions[i].name, p->target_name) == 0) {
                addr = c->functions[i].address;
                break;
            }
        }
        if (addr == 0xFFFF) {
            CODEGEN_ERROR("Unresolved function reference to %s", p->target_name);
        }
        bb_patch_u16(&c->code, p->addr_offset, addr);
    }
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Helpers to emit expressions                                                */
/* ──────────────────────────────────────────────────────────────────────────── */

static void compile_identifier(Compiler *c, const char *name) {
    int local_idx = map_lookup(&c->locals, name);
    if (local_idx >= 0) {
        bb_write_u8(&c->code, OP_LOAD_LOCAL);
        bb_write_u8(&c->code, (uint8_t)local_idx);
        return;
    }
    int global_idx = map_lookup(&c->globals, name);
    if (global_idx < 0) {
        CODEGEN_ERROR("Use of undeclared variable '%s'", name);
    }
    bb_write_u8(&c->code, OP_LOAD_GLOBAL);
    bb_write_u8(&c->code, (uint8_t)global_idx);
}

static void compile_literal(Compiler *c, ASTNode *n) {
    switch (n->type) {
        case AST_INT_LITERAL:
            bb_write_u8(&c->code, OP_PUSH_I32);
            bb_write_u32(&c->code, (uint32_t)n->as.int_val);
            break;
        case AST_FLOAT_LITERAL: {
            union { float f; uint32_t u; } conv;
            conv.f = n->as.float_val;
            bb_write_u8(&c->code, OP_PUSH_F32);
            bb_write_u32(&c->code, conv.u);
            break; }
        case AST_BOOL_LITERAL:
            bb_write_u8(&c->code, n->as.bool_val ? OP_PUSH_TRUE : OP_PUSH_FALSE);
            break;
        case AST_NIL_LITERAL:
            bb_write_u8(&c->code, OP_PUSH_NIL);
            break;
        default:
            CODEGEN_ERROR("Unexpected node in compile_literal");
    }
}

static uint8_t opcode_for_binary(TokenType op) {
    switch (op) {
        case TOK_PLUS:          return OP_ADD;
        case TOK_MINUS:         return OP_SUB;
        case TOK_STAR:          return OP_MUL;
        case TOK_SLASH:         return OP_DIV;
        case TOK_PERCENT:       return OP_MOD;
        case TOK_LESS:          return OP_LT;
        case TOK_LESS_EQUAL:    return OP_LE;
        case TOK_GREATER:       return OP_GT;
        case TOK_GREATER_EQUAL: return OP_GE;
        case TOK_EQUAL_EQUAL:   return OP_EQ;
        case TOK_BANG_EQUAL:    return OP_NEQ;
        case TOK_AND_AND:       return OP_AND;
        case TOK_OR_OR:         return OP_OR;
        default:                return 0xFF;
    }
}

static uint8_t opcode_for_unary(TokenType op) {
    switch (op) {
        case TOK_MINUS: return OP_NEG;
        case TOK_BANG:  return OP_NOT;
        default:        return 0xFF;
    }
}

static void compile_expression(Compiler *c, ASTNode *expr) {
    if (!expr) return;
    switch (expr->type) {
        case AST_INT_LITERAL:
        case AST_FLOAT_LITERAL:
        case AST_BOOL_LITERAL:
        case AST_NIL_LITERAL:
            compile_literal(c, expr);
            break;
        case AST_IDENTIFIER:
            compile_identifier(c, expr->as.identifier);
            break;
        case AST_UNARY: {
            compile_expression(c, expr->as.unary.expr);
            uint8_t op = opcode_for_unary(expr->as.unary.op);
            if (op == 0xFF) CODEGEN_ERROR("Unsupported unary op");
            bb_write_u8(&c->code, op);
            break; }
        case AST_BINARY: {
            /* left then right so that right ends up on top of stack */
            compile_expression(c, expr->as.binary.left);
            compile_expression(c, expr->as.binary.right);
            uint8_t op = opcode_for_binary(expr->as.binary.op);
            if (op == 0xFF) CODEGEN_ERROR("Unsupported binary op");
            bb_write_u8(&c->code, op);
            break; }
        case AST_FUNCTION_CALL: {
            /* arguments */
            uint8_t argc = 0;
            ASTNode *arg = expr->as.func_call.args;
            while (arg) {
                compile_expression(c, arg);
                argc++;
                arg = arg->next;
            }
            /* emit call */
            size_t addr_off = bb_write_u8(&c->code, OP_CALL);
            addr_off = bb_write_u16(&c->code, 0xFFFF); /* patched later */
            bb_write_u8(&c->code, argc);
            register_patch(c, addr_off, expr->as.func_call.name);
            break; }
        case AST_NATIVE_CALL: {
            /* arguments */
            uint8_t argc = 0;
            ASTNode *arg = expr->as.native_call.args;
            while (arg) {
                compile_expression(c, arg);
                argc++;
                arg = arg->next;
            }
            int nat_idx = -1;
            for (uint8_t i = 0; i < c->device_tbl->native_count; ++i) {
                if (strcmp(c->device_tbl->names[i], expr->as.native_call.name) == 0) {
                    nat_idx = i;
                    break;
                }
            }
            if (nat_idx < 0) {
                CODEGEN_ERROR("Unknown native '%s' for device %u", expr->as.native_call.name, c->device_tbl->device_id);
            }
            bb_write_u8(&c->code, OP_NCALL);
            bb_write_u8(&c->code, (uint8_t)nat_idx);
            bb_write_u8(&c->code, argc);
            break; }
        default:
            CODEGEN_ERROR("Expression type %d not implemented", expr->type);
    }
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Statement / block compilation                                              */
/* ──────────────────────────────────────────────────────────────────────────── */

static void compile_assignment(Compiler *c, ASTNode *stmt) {
    if (stmt->as.assign.is_local) {
        /* declare if first time */
        int idx = map_lookup(&c->locals, stmt->as.assign.name);
        if (idx < 0) {
            CODEGEN_ERROR("Use of undeclared variable '%s'", stmt->as.assign.name);
        }

        /* now compile the value expression */
        compile_expression(c, stmt->as.assign.value);

        /* and store the result */
        bb_write_u8(&c->code, OP_STORE_LOCAL);
        bb_write_u8(&c->code, (uint8_t)idx);
    } else {
        /* globals are simpler – expression first, then store */
        compile_expression(c, stmt->as.assign.value);
        int idx = map_lookup(&c->globals, stmt->as.assign.name);
        if (idx < 0) idx = map_insert(&c->globals, stmt->as.assign.name, (int)c->globals.count);
        bb_write_u8(&c->code, OP_STORE_GLOBAL);
        bb_write_u8(&c->code, (uint8_t)idx);
    }
}

static void compile_if(Compiler *c, ASTNode *stmt) {
    compile_expression(c, stmt->as.if_stmt.condition);
    /* emit conditional jump over THEN if false */
    bb_write_u8(&c->code, OP_JMP_IF_FALSE);
    size_t jmp_false_off = bb_write_u16(&c->code, 0xFFFF);

    compile_block(c, stmt->as.if_stmt.then_body, false);

    if (stmt->as.if_stmt.else_body) {
        /* Unconditional jump to skip ELSE branch after THEN executes */
        bb_write_u8(&c->code, OP_JMP);
        size_t jmp_end_off = bb_write_u16(&c->code, 0xFFFF);

        /* Patch the false‐jump to point to the start of the ELSE branch */
        uint16_t else_addr = (uint16_t)c->code.len;
        int16_t  rel_else  = (int16_t)(else_addr - (jmp_false_off + 2));
        bb_patch_u16(&c->code, jmp_false_off, (uint16_t)rel_else);

        /* Emit ELSE branch – can be either a block or another if (elif) */
        if (stmt->as.if_stmt.else_body->type == AST_BLOCK) {
            compile_block(c, stmt->as.if_stmt.else_body, false);
        } else if (stmt->as.if_stmt.else_body->type == AST_IF) {
            compile_if(c, stmt->as.if_stmt.else_body);
        } else {
            CODEGEN_ERROR("Else body must be a block or if statement");
        }

        /* Patch end jump */
        uint16_t end_addr = (uint16_t)c->code.len;
        int16_t  rel_end  = (int16_t)(end_addr - (jmp_end_off + 2));
        bb_patch_u16(&c->code, jmp_end_off, (uint16_t)rel_end);
    } else {
        /* No ELSE – point the false jump to after THEN */
        uint16_t after_then = (uint16_t)c->code.len;
        int16_t  rel_after  = (int16_t)(after_then - (jmp_false_off + 2));
        bb_patch_u16(&c->code, jmp_false_off, (uint16_t)rel_after);
    }
}

static void compile_while(Compiler *c, ASTNode *stmt) {
    uint16_t loop_start = (uint16_t)c->code.len;
    compile_expression(c, stmt->as.while_stmt.condition);
    bb_write_u8(&c->code, OP_JMP_IF_FALSE);
    size_t exit_patch_off = bb_write_u16(&c->code, 0xFFFF);

    compile_block(c, stmt->as.while_stmt.body, false);
    /* jump back (relative) */
    bb_write_u8(&c->code, OP_JMP);
    int16_t back_offset = (int16_t)(loop_start - ((uint16_t)c->code.len + 2));
    bb_write_u16(&c->code, (uint16_t)back_offset);

    /* patch exit to point after loop */
    uint16_t after_loop = (uint16_t)c->code.len;
    int16_t rel_exit = (int16_t)(after_loop - (exit_patch_off + 2));
    bb_patch_u16(&c->code, exit_patch_off, (uint16_t)rel_exit);
}

static void compile_statement(Compiler *c, ASTNode *stmt) {
    switch (stmt->type) {
        case AST_ASSIGNMENT:
            compile_assignment(c, stmt);
            break;
        case AST_NATIVE_CALL:
        case AST_FUNCTION_CALL:
            compile_expression(c, stmt); /* produce call */
            bb_write_u8(&c->code, OP_POP); /* discard result */
            break;
        case AST_WHILE:
            compile_while(c, stmt);
            break;
        case AST_IF:
            compile_if(c, stmt);
            break;
        case AST_BLOCK:
            compile_block(c, stmt, false);
            break;
        case AST_RETURN:
            compile_expression(c, stmt->as.return_stmt.value);
            break;
        default:
            /* expression statement */
            compile_expression(c, stmt);
            bb_write_u8(&c->code, OP_POP);
            break;
    }
}

static void compile_block(Compiler *c, ASTNode *block, bool is_func_body) {
    if (!block) return;
    assert(block->type == AST_BLOCK);

    /* Track current local count so we can pop at the end if required */
    uint8_t start_local_count = c->local_count;

    /* ── Pass 1: scan for new local declarations so we can reserve slots        */
    ASTNode *scan = block->as.block.stmts;
    while (scan) {
        if (scan->type == AST_ASSIGNMENT && scan->as.assign.is_local) {
            int idx = map_lookup(&c->locals, scan->as.assign.name);
            if (idx < 0) {
                idx = map_insert(&c->locals, scan->as.assign.name, c->local_count++);
                /* Reserve a slot on the VM stack – keeps call-frame layout stable */
                bb_write_u8(&c->code, OP_PUSH_NIL);
            }
        }
        scan = scan->next;
    }

    /* ── Pass 2: emit byte-code for every statement in order                    */
    ASTNode *cur = block->as.block.stmts;
    while (cur) {
        compile_statement(c, cur);
        cur = cur->next;
    }

    /* ── Pass 3: pop locals introduced in this block (except for function body) */
    if (!is_func_body) {
        uint8_t locals_added = c->local_count - start_local_count;
        for (uint8_t i = 0; i < locals_added; ++i) {
            bb_write_u8(&c->code, OP_POP);
        }
        /* Remove locals from bookkeeping */
        if (locals_added > 0) {
            for (size_t i = 0; i < c->locals.cap; ++i) {
                if (c->locals.entries && c->locals.entries[i].key &&
                    c->locals.entries[i].value >= start_local_count) {
                    free(c->locals.entries[i].key);
                    c->locals.entries[i].key = NULL;
                }
            }
            c->local_count = start_local_count;
        }
    }
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Pre-scan to register global variables                                     */
/* ──────────────────────────────────────────────────────────────────────────── */

static void prescan_globals_in_stmt(Compiler *c, ASTNode *stmt);

static void prescan_globals_in_block(Compiler *c, ASTNode *block) {
    if (!block || block->type != AST_BLOCK) return;
    ASTNode *stmt = block->as.block.stmts;
    while (stmt) {
        prescan_globals_in_stmt(c, stmt);
        stmt = stmt->next;
    }
}

static void prescan_globals_in_stmt(Compiler *c, ASTNode *stmt) {
    if (!stmt) return;

    switch (stmt->type) {
        case AST_ASSIGNMENT:
            if (!stmt->as.assign.is_local) {
                /* Register global if not already present */
                int idx = map_lookup(&c->globals, stmt->as.assign.name);
                if (idx < 0) {
                    map_insert(&c->globals, stmt->as.assign.name, (int)c->globals.count);
                }
            }
            break;
        case AST_WHILE:
            prescan_globals_in_block(c, stmt->as.while_stmt.body);
            break;
        case AST_IF:
            prescan_globals_in_block(c, stmt->as.if_stmt.then_body);
            if (stmt->as.if_stmt.else_body) {
                if (stmt->as.if_stmt.else_body->type == AST_BLOCK) {
                    prescan_globals_in_block(c, stmt->as.if_stmt.else_body);
                } else if (stmt->as.if_stmt.else_body->type == AST_IF) {
                    prescan_globals_in_stmt(c, stmt->as.if_stmt.else_body);
                }
            }
            break;
        case AST_BLOCK:
            prescan_globals_in_block(c, stmt);
            break;
        default:
            /* Other statements don't introduce globals */
            break;
    }
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Function definition                                                        */
/* ──────────────────────────────────────────────────────────────────────────── */

static void ensure_func_cap(Compiler *c) {
    if (c->func_count == c->func_cap) {
        c->func_cap = c->func_cap ? c->func_cap * 2 : 8;
        c->functions = realloc(c->functions, c->func_cap * sizeof(struct FunctionInfo));
        if (!c->functions) { fprintf(stderr, "OOM functions\n"); exit(1);}    }
}

static void compile_function_def(Compiler *c, ASTNode *func_def) {
    ensure_func_cap(c);
    size_t idx = c->func_count++;
    c->functions[idx].name = func_def->as.func_def.name;
    c->functions[idx].address = (uint16_t)(c->code.len - c->code_base);
    /* count params */
    uint8_t paramc = 0; ASTNode *p = func_def->as.func_def.params; while (p) { paramc++; p = p->next; }
    c->functions[idx].arity = paramc;

    /* set up new locals map (parameters are locals 0..n-1) */
    StrIntMap old_locals = c->locals;
    uint8_t   old_local_count = c->local_count;
    map_init(&c->locals);
    c->local_count = 0;
    /* add params */
    p = func_def->as.func_def.params;
    while (p) {
        map_insert(&c->locals, p->as.param.param_name, c->local_count++);
        p = p->next;
    }

    compile_block(c, func_def->as.func_def.body, true);

    /* Ensure function always returns (push nil) */
    ASTNode *last_stmt = NULL;
    ASTNode *stmt = func_def->as.func_def.body->as.block.stmts;
    while (stmt) {
        last_stmt = stmt;
        stmt = stmt->next;
    }
    if (!last_stmt || last_stmt->type != AST_RETURN) {
        bb_write_u8(&c->code, OP_PUSH_NIL);
    }
    bb_write_u8(&c->code, OP_RET);
    bb_write_u8(&c->code, 1); /* return value count */

    /* restore locals */
    /* free param keys */
    for (size_t i = 0; i < c->locals.cap; i++) if (c->locals.entries && c->locals.entries[i].key) free(c->locals.entries[i].key);
    free(c->locals.entries);
    c->locals = old_locals;
    c->local_count = old_local_count;
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Public entry point                                                         */
/* ──────────────────────────────────────────────────────────────────────────── */

int smollu_generate_bytecode(ASTNode        *root,
                              uint8_t         device_id,
                              uint8_t         version,
                              uint8_t       **out_buf,
                              size_t         *out_len) {
    if (!root || root->type != AST_PROGRAM) {
        CODEGEN_ERROR("Root node is not AST_PROGRAM");
    }

    Compiler c;
    memset(&c, 0, sizeof(c));
    bb_init(&c.code);
    map_init(&c.globals);
    map_init(&c.locals);

    /* ── Resolve device native table */
    c.device_tbl = smollu_get_device_native_table(device_id);
    if (!c.device_tbl) {
        CODEGEN_ERROR("Unknown device id %u", device_id);
    }

    /* ── Reserve space for header (16 bytes) – will fill later */
    for (int i = 0; i < 16; i++) bb_write_u8(&c.code, 0x00);

    /* ── Append native table (2 bytes per entry) */
    for (size_t i = 0; i < c.device_tbl->native_count; i++) {
        bb_write_u16(&c.code, (uint16_t)i);
    }

    /* Record where the code segment actually begins */
    size_t code_base = c.code.len; /* this will be address 0 for the VM */
    c.code_base = code_base;

    /* Reserve a JMP (+offset int16) at address 0 (start of code) */
    size_t jmp_placeholder_offset = c.code.len; /* should equal code_base */
    bb_write_u8(&c.code, OP_JMP);
    size_t jmp_offset_field = bb_write_u16(&c.code, 0xFFFF);

    /* ── Pre-scan init and main blocks to register all global variables */
    prescan_globals_in_block(&c, root->as.program.init);
    prescan_globals_in_block(&c, root->as.program.main);

    /* ── First compile all user functions so that their addresses are known */
    ASTNode *func_iter = root->as.program.functions->as.block.stmts;
    while (func_iter) {
        if (func_iter->type != AST_FUNCTION_DEF) {
            CODEGEN_ERROR("Non-function node found in functions block");
        }
        compile_function_def(&c, func_iter);
        func_iter = func_iter->next;
    }

    /* Record relative address of init block */
    uint16_t init_addr_rel = (uint16_t)(c.code.len - c.code_base);

    /* ── Compile init and main blocks (in that order) */
    compile_block(&c, root->as.program.init, false);
    compile_block(&c, root->as.program.main, false);

    /* ── Patch the jump at code start to jump to init */
    int16_t jmp_rel = (int16_t)(init_addr_rel - 3); /* pc after offset = 3 */
    bb_patch_u16(&c.code, jmp_offset_field, (uint16_t)jmp_rel);

    /* program end / safety halt */
    bb_write_u8(&c.code, OP_HALT);

    /* ── Patch pending call addresses */
    resolve_patches(&c);

    size_t code_size = c.code.len - 16 - 2 * c.device_tbl->native_count; /* code bytes so far (excluding header and native table) */

    /* ── Fill header fields */
    const char *magic = "SMOL";
    memcpy(&c.code.data[0], magic, 4);
    c.code.data[4] = version;
    c.code.data[5] = device_id;
    c.code.data[6] = (uint8_t)c.func_count;
    c.code.data[7] = c.device_tbl->native_count;
    bb_patch_u32(&c.code, 8, (uint32_t)code_size); /* code_size excludes native table */
    /* reserved stays 0 */

    /* output */
    *out_buf = c.code.data;
    *out_len = c.code.len;
    
    /* don't free buffer – ownership transferred */
    return 0;
}

/* helper to patch 32-bit little-endian */
static void bb_patch_u32(ByteBuf *bb, size_t at, uint32_t v) {
    assert(at + 3 < bb->len);
    for (int i = 0; i < 4; i++) bb->data[at + i] = (v >> (8*i)) & 0xFF;
}

