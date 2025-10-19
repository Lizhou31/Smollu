#include "smollu_vm.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#define FLOAT_EPSILON 0.00001f

/* ────────────────────────────────────────────────────────────────────────── */
/*  Helpers                                                                   */
/* ────────────────────────────────────────────────────────────────────────── */

static inline void push(SmolluVM *vm, Value v) {
    assert(vm->sp < SMOLLU_STACK_MAX - 1 && "Stack overflow");
    vm->stack[vm->sp++] = v;
}

static inline Value pop(SmolluVM *vm) {
    assert(vm->sp > 0 && "Stack underflow");
    return vm->stack[--vm->sp];
}

static inline Value peek(SmolluVM *vm, uint8_t distance) {
    assert(distance < vm->sp);
    return vm->stack[vm->sp - 1 - distance];
}

#ifdef CONFIG_SMOLLU_STREAMING
/**
 * @brief Reload cache with bytecode from streaming source
 */
static int reload_cache(SmolluVM *vm, uint32_t pc) {
    if (!vm->read_region || !vm->stream_ctx) {
        fprintf(stderr, "Stream not configured\n");
        return -1;
    }

    /* Calculate offset in source (code starts at code_offset) */
    uint32_t source_offset = vm->code_offset + pc;

    /* Calculate how many bytes to read (don't exceed code size) */
    uint32_t remaining = vm->bc_len - pc;
    uint16_t read_size = (remaining < SMOLLU_STREAM_CACHE_SIZE) ?
                         remaining : SMOLLU_STREAM_CACHE_SIZE;

    if (read_size == 0) {
        fprintf(stderr, "Cache reload: no data to read at pc=%u\n", pc);
        return -1;
    }

    /* Read from source */
    int ret = vm->read_region(vm->stream_ctx, source_offset, vm->cache, read_size);
    if (ret != 0) {
        fprintf(stderr, "Cache reload failed at offset %u: %d\n", source_offset, ret);
        return ret;
    }

    /* Update cache metadata */
    vm->cache_start = pc;
    vm->cache_len = read_size;

    return 0;
}
#endif /* CONFIG_SMOLLU_STREAMING */

/**
 * @brief Fetch byte from bytecode (streaming or in-RAM)
 */
static inline uint8_t fetch_u8(SmolluVM *vm) {
#ifdef CONFIG_SMOLLU_STREAMING
    /* Streaming mode: bytecode pointer is NULL */
    if (vm->bytecode == NULL) {
        /* Check if PC is within cached region */
        if (vm->pc < vm->cache_start || vm->pc >= vm->cache_start + vm->cache_len) {
            if (reload_cache(vm, vm->pc) != 0) {
                fprintf(stderr, "Fatal: cache reload failed at pc=%zu\n", vm->pc);
                return 0xFF;  /* Return illegal opcode on error */
            }
        }
        return vm->cache[vm->pc++ - vm->cache_start];
    }
#endif

    /* Legacy in-RAM mode */
    assert(vm->pc < vm->bc_len);
    return vm->bytecode[vm->pc++];
}

static inline uint8_t read_u8(SmolluVM *vm) {
    return fetch_u8(vm);
}

static inline int8_t read_i8(SmolluVM *vm) {
    return (int8_t)read_u8(vm);
}

static inline uint16_t read_u16(SmolluVM *vm) {
    uint16_t lo = read_u8(vm);
    uint16_t hi = read_u8(vm);
    return lo | (hi << 8);
}

static inline int16_t read_i16(SmolluVM *vm) {
    return (int16_t)read_u16(vm);
}


/* ────────────────────────────────────────────────────────────────────────── */
/*  Opcode definitions                                                        */
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

/* ────────────────────────────────────────────────────────────────────────── */
/*  Runtime utilities                                                         */
/* ────────────────────────────────────────────────────────────────────────── */

static Value arithmetic(Value a, Value b, uint8_t op) {
    if (a.type == VAL_INT && b.type == VAL_INT) {
        switch (op) {
            case OP_ADD: return value_from_int(b.as.i + a.as.i);
            case OP_SUB: return value_from_int(b.as.i - a.as.i);
            case OP_MUL: return value_from_int(b.as.i * a.as.i);
            case OP_DIV: return value_from_int(a.as.i == 0 ? 0 : b.as.i / a.as.i);
            case OP_MOD: return value_from_int(a.as.i == 0 ? 0 : b.as.i % a.as.i);
            default: break;
        }
    } else {
        /* Coerce to float */
        float af = (a.type == VAL_FLOAT) ? a.as.f : (float)a.as.i;
        float bf = (b.type == VAL_FLOAT) ? b.as.f : (float)b.as.i;
        switch (op) {
            case OP_ADD: return value_from_float(bf + af);
            case OP_SUB: return value_from_float(bf - af);
            case OP_MUL: return value_from_float(bf * af);
            case OP_DIV: return value_from_float(af == 0.0f ? 0.0f : bf / af);
            default: break;
        }
    }
    return value_make_nil();
}

static Value compare(Value a, Value b, uint8_t op) {
    if (a.type == VAL_INT && b.type == VAL_INT) {
        switch (op) {
            case OP_EQ:  return value_from_bool(b.as.i == a.as.i);
            case OP_NEQ: return value_from_bool(b.as.i != a.as.i);
            case OP_LT:  return value_from_bool(b.as.i <  a.as.i);
            case OP_LE:  return value_from_bool(b.as.i <= a.as.i);
            case OP_GT:  return value_from_bool(b.as.i >  a.as.i);
            case OP_GE:  return value_from_bool(b.as.i >= a.as.i);
        }
    } else {
        float af = (a.type == VAL_FLOAT) ? a.as.f : (float)a.as.i;
        float bf = (b.type == VAL_FLOAT) ? b.as.f : (float)b.as.i;
        switch (op) {
            case OP_EQ:  return value_from_bool(fabsf(bf - af) < FLOAT_EPSILON);
            case OP_NEQ: return value_from_bool(fabsf(bf - af) >= FLOAT_EPSILON);
            case OP_LT:  return value_from_bool(bf < af);
            case OP_LE:  return value_from_bool(bf < af || fabsf(bf - af) < FLOAT_EPSILON);
            case OP_GT:  return value_from_bool(bf > af);
            case OP_GE:  return value_from_bool(bf > af || fabsf(bf - af) < FLOAT_EPSILON);
        }
    }
    return value_make_nil();
}

static void smollu_vm_read_header(SmolluVM *vm, const uint8_t *bytecode) {
    vm->magic = bytecode[0] | (bytecode[1] << 8) | (bytecode[2] << 16) | (bytecode[3] << 24);
    vm->version = bytecode[4];
    vm->device_id = bytecode[5];
    vm->native_count = bytecode[7];
    vm->code_size = bytecode[8] | (bytecode[9] << 8) | (bytecode[10] << 16) | (bytecode[11] << 24);
    vm->reserved = bytecode[12] | (bytecode[13] << 8) | (bytecode[14] << 16) | (bytecode[15] << 24);
}

static void smollu_vm_read_native_table(SmolluVM *vm, const uint8_t *bytecode, const NativeFn *native_table) {
    for (uint8_t i = 0; i < vm->native_count; ++i) {
        vm->natives[i] = native_table[bytecode[i * 2] | (bytecode[i * 2 + 1] << 8)];
    }
}

static void smollu_vm_read_code(SmolluVM *vm, const uint8_t *bytecode, size_t len) {
    vm->bytecode = bytecode;
    vm->bc_len   = len;
    vm->pc       = 0;
    vm->sp       = 0;
    vm->fp       = 0;
}

/* ────────────────────────────────────────────────────────────────────────── */
/*  Public API                                                                */
/* ────────────────────────────────────────────────────────────────────────── */

void smollu_vm_init(SmolluVM *vm) {
    memset(vm, 0, sizeof(*vm));
}

void smollu_vm_prepare(SmolluVM *vm, const uint8_t *bytecode, const NativeFn *native_table) {
    smollu_vm_read_header(vm, bytecode);
    if (native_table) {
        smollu_vm_read_native_table(vm, bytecode + 16, native_table);
    }
}

void smollu_vm_load(SmolluVM *vm, const uint8_t *bytecode, size_t len) {
    smollu_vm_read_code(vm, bytecode, len);
}

void smollu_vm_register_native(SmolluVM *vm, uint8_t nat_id, NativeFn fn) {
    vm->natives[nat_id] = fn;
}

void smollu_vm_destroy(SmolluVM *vm) {
    (void)vm; /* nothing for now */
}

#ifdef CONFIG_SMOLLU_STREAMING
/**
 * @brief Prepare VM for streaming execution from external source
 */
int smollu_vm_prepare_streaming(SmolluVM *vm,
                                int (*read_region)(void *ctx, uint32_t offset, uint8_t *dst, size_t len),
                                void *ctx,
                                const NativeFn *native_table)
{
    if (!vm || !read_region || !ctx) {
        fprintf(stderr, "Invalid parameters for streaming\n");
        return -1;
    }

    /* Temporarily set up read_region for header/native table reading */
    vm->stream_ctx = ctx;
    vm->read_region = read_region;

    /* Read 16-byte header */
    uint8_t header[16];
    if (read_region(ctx, 0, header, 16) != 0) {
        fprintf(stderr, "Failed to read bytecode header\n");
        return -1;
    }

    /* Parse header */
    smollu_vm_read_header(vm, header);

    /* Validate magic number */
    if (vm->magic != 0x4C4F4D53) {  /* "SMOL" in little-endian */
        fprintf(stderr, "Invalid bytecode magic: 0x%08X\n", vm->magic);
        return -1;
    }

    /* Read native function table if present */
    if (vm->native_count > 0) {
        if (!native_table) {
            fprintf(stderr, "Native functions required but table not provided\n");
            return -1;
        }

        /* Read native table (2 bytes per entry) */
        uint8_t native_buf[512];  /* Max 256 entries * 2 bytes */
        size_t native_size = vm->native_count * 2;

        if (native_size > sizeof(native_buf)) {
            fprintf(stderr, "Native table too large: %zu bytes\n", native_size);
            return -1;
        }

        if (read_region(ctx, 16, native_buf, native_size) != 0) {
            fprintf(stderr, "Failed to read native function table\n");
            return -1;
        }

        /* Map native functions */
        smollu_vm_read_native_table(vm, native_buf, native_table);
    }

    /* Calculate code offset: header (16) + native table (native_count * 2) */
    vm->code_offset = 16 + (vm->native_count * 2);

    /* Setup streaming mode */
    vm->bytecode = NULL;           /* NULL indicates streaming mode */
    vm->bc_len = vm->code_size;    /* Total code size from header */
    vm->pc = 0;
    vm->sp = 0;
    vm->fp = 0;

    /* Initialize cache as empty */
    vm->cache_start = 0;
    vm->cache_len = 0;

    printf("[VM] Streaming mode initialized: code_size=%u, code_offset=%u, native_count=%u\n",
           vm->code_size, vm->code_offset, vm->native_count);

    return 0;
}
#endif

/* Return value: 0 on success, non-zero on runtime error */
int smollu_vm_run(SmolluVM *vm) {
    for (;;) {
        if (vm->pc >= vm->bc_len) {
            fprintf(stderr, "PC out of bounds!\n");
            return -1;
        }
        uint8_t op = read_u8(vm);
        switch (op) {
            /* 00–0F */
            case OP_NOP: break;
            case OP_PUSH_NIL:  push(vm, value_make_nil()); break;
            case OP_PUSH_TRUE: push(vm, value_from_bool(true)); break;
            case OP_PUSH_FALSE:push(vm, value_from_bool(false)); break;
            case OP_PUSH_I8:   push(vm, value_from_int(read_i8(vm))); break;
            case OP_PUSH_I32: {
                int32_t val = (int32_t)read_u16(vm) | ((int32_t)read_u16(vm) << 16);
                push(vm, value_from_int(val));
                break;
            }
            case OP_PUSH_F32: {
                /* Read IEEE 754 little-endian */
                uint32_t bits = read_u16(vm) | ((uint32_t)read_u16(vm) << 16);
                float f;
                memcpy(&f, &bits, sizeof(f));
                push(vm, value_from_float(f));
                break;
            }
            case OP_DUP:   push(vm, peek(vm, 0)); break;
            case OP_POP:   pop(vm);               break;
            case OP_SWAP: {
                Value a = pop(vm);
                Value b = pop(vm);
                push(vm, a);
                push(vm, b);
                break;
            }

            /* Locals / globals – locals are addressed relative to current frame base */
            case OP_LOAD_LOCAL: {
                uint8_t slot = read_u8(vm);
                uint8_t base = vm->fp ? vm->frames[vm->fp - 1].base : 0;
                push(vm, vm->stack[base + slot]);
                break;
            }
            case OP_STORE_LOCAL: {
                uint8_t slot = read_u8(vm);
                Value v = pop(vm);
                uint8_t base = vm->fp ? vm->frames[vm->fp - 1].base : 0;
                vm->stack[base + slot] = v;
                break;
            }
            case OP_LOAD_GLOBAL: {
                uint8_t slot = read_u8(vm);
                push(vm, vm->globals[slot]);
                break;
            }
            case OP_STORE_GLOBAL: {
                uint8_t slot = read_u8(vm);
                vm->globals[slot] = pop(vm);
                break;
            }

            /* Arithmetic */
            case OP_ADD: case OP_SUB: case OP_MUL: case OP_DIV: case OP_MOD: {
                Value a = pop(vm);
                Value b = pop(vm);
                push(vm, arithmetic(a, b, op));
                break;
            }
            case OP_NEG: {
                Value a = pop(vm);
                if (a.type == VAL_INT) a.as.i = -a.as.i;
                else if (a.type == VAL_FLOAT) a.as.f = -a.as.f;
                push(vm, a);
                break;
            }
            case OP_NOT: {
                Value a = pop(vm);
                bool truthy = false;
                switch (a.type) {
                    case VAL_NIL:   truthy = false; break;
                    case VAL_BOOL:  truthy = a.as.boolean; break;
                    case VAL_INT:   truthy = a.as.i != 0; break;
                    case VAL_FLOAT: truthy = a.as.f != 0.0f; break;
                }
                push(vm, value_from_bool(!truthy));
                break;
            }
            case OP_AND: {
                Value a = pop(vm);
                Value b = pop(vm);
                push(vm, value_from_bool(b.as.i != 0 && a.as.i != 0));
                break;
            }
            case OP_OR: {
                Value a = pop(vm);
                Value b = pop(vm);
                push(vm, value_from_bool(b.as.i != 0 || a.as.i != 0));
                break;
            }
            /* Comparison */
            case OP_EQ: case OP_NEQ: case OP_LT: case OP_LE: case OP_GT: case OP_GE: {
                Value a = pop(vm);
                Value b = pop(vm);
                push(vm, compare(a, b, op));
                break;
            }

            /* Control flow */
            case OP_JMP: {
                int16_t off = read_i16(vm);
                vm->pc += off;
                break;
            }
            case OP_JMP_IF_TRUE: {
                int16_t off = read_i16(vm);
                Value cond = pop(vm);
                bool truthy = false;
                if (cond.type == VAL_BOOL) truthy = cond.as.boolean;
                else if (cond.type == VAL_INT) truthy = cond.as.i != 0;
                else if (cond.type == VAL_FLOAT) truthy = cond.as.f != 0.0f;
                if (truthy) vm->pc += off;
                break;
            }
            case OP_JMP_IF_FALSE: {
                int16_t off = read_i16(vm);
                Value cond = pop(vm);
                bool truthy = false;
                if (cond.type == VAL_BOOL) truthy = cond.as.boolean;
                else if (cond.type == VAL_INT) truthy = cond.as.i != 0;
                else if (cond.type == VAL_FLOAT) truthy = cond.as.f != 0.0f;
                if (!truthy) vm->pc += off;
                break;
            }
            case OP_HALT:
                return 0; /* Normal termination */

            /* Function */
            case OP_CALL: {
                uint16_t func_addr = read_u16(vm);
                uint8_t argc      = read_u8(vm);
                if (vm->fp >= SMOLLU_CALLSTACK_MAX) {
                    fprintf(stderr,"Call stack overflow\n");
                    return -1;
                }
                vm->frames[vm->fp].ret_pc = (uint32_t)vm->pc;
                vm->frames[vm->fp].base   = vm->sp - argc;
                vm->fp++;
                vm->pc = func_addr;
                break;
            }
            case OP_RET: {
                uint8_t rv_cnt = read_u8(vm);
                /* Pop return values to temp */
                Value retvals[10]; /* up to 10 returns */
                if (rv_cnt > 10) rv_cnt = 10;
                for (uint8_t i = 0; i < rv_cnt; ++i) {
                    retvals[i] = pop(vm);
                }
                /* Discard locals by resetting sp to base */
                if (vm->fp == 0) {
                    fprintf(stderr,"Return with empty call stack\n");
                    return -1;
                }
                vm->fp--;
                vm->sp = vm->frames[vm->fp].base;
                vm->pc = vm->frames[vm->fp].ret_pc;
                /* Push return values back (in original order) */
                for (int i = rv_cnt - 1; i >= 0; --i) {
                    push(vm, retvals[i]);
                }
                break;
            }

            /* Native call */
            case OP_NCALL: {
                uint8_t nat_id = read_u8(vm);
                uint8_t argc   = read_u8(vm);
                NativeFn fn = vm->natives[nat_id];
                if (!fn) {
                    fprintf(stderr,"Unknown native %u\n", nat_id);
                    return -1;
                }
                Value *args = &vm->stack[vm->sp - argc];
                Value res = fn(args, argc);
                vm->sp -= argc;
                push(vm, res);
                break;
            }
            case OP_SLEEP_MS: {
                uint16_t ms = read_u16(vm);
                /* Simple busy wait – not ideal but placeholder */
                for (volatile uint32_t i = 0; i < (uint32_t)ms * 1000; ++i) {
                    /* nop */
                }
                break;
            }

            /* Meta */
            case OP_DBG_TRAP: {
                printf("[VM] DEBUG TRAP at pc=%zu\n", vm->pc);
                break;
            }
            case OP_ILLEGAL:
            default:
                fprintf(stderr, "Illegal opcode 0x%02X at pc=%zu\n", op, vm->pc-1);
                return -1;
        }
    }
}
