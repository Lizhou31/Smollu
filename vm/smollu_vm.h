/**
 * @file smollu_vm.h
 * @author Lizhou (lisie31s@gmail.com)
 * @brief smollu vm header file
 * 
 * @version 0.1 
 * @date 2025-07-29
 * 
 * @copyright Copyright (c) 2025 Lizhou
 * 
 */

#ifndef SMOLLU_VM_H
#define SMOLLU_VM_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ────────────────────────────────────────────────────────────────────────── */
/*  Public Value type                                                        */
/* ────────────────────────────────────────────────────────────────────────── */

typedef enum {
    VAL_NIL,
    VAL_BOOL,
    VAL_INT,
    VAL_FLOAT,
} ValueType;

typedef struct {
    ValueType type;
    union {
        bool    boolean;
        int32_t i;
        float   f;
    } as;
} Value;

static inline Value value_make_nil(void)           { return (Value){ .type = VAL_NIL   }; }
static inline Value value_from_bool(bool b)        { return (Value){ .type = VAL_BOOL  , .as.boolean = b }; }
static inline Value value_from_int(int32_t i)      { return (Value){ .type = VAL_INT   , .as.i = i }; }
static inline Value value_from_float(float f)      { return (Value){ .type = VAL_FLOAT , .as.f = f }; }

/* ────────────────────────────────────────────────────────────────────────── */
/*  Native Function Interface                                                */
/* ────────────────────────────────────────────────────────────────────────── */

typedef Value (*NativeFn)(Value *args, uint8_t argc);

/* ────────────────────────────────────────────────────────────────────────── */
/*  VM Object                                                                */
/* ────────────────────────────────────────────────────────────────────────── */

#define SMOLLU_STACK_MAX 255
#define SMOLLU_CALLSTACK_MAX 64

typedef struct {
    uint32_t    ret_pc;
    uint8_t     base;      /* base index in the value stack */
} CallFrame;

typedef struct {
    /* Program */
    const uint8_t *bytecode;
    size_t         bc_len;

    /* Header */
    uint32_t magic;
    uint8_t  version;
    uint8_t  device_id;
    uint8_t  native_count;
    uint32_t code_size;
    uint32_t reserved;

    /* Execution state */
    size_t   pc;                   /* program counter */
    Value    stack[SMOLLU_STACK_MAX];
    uint8_t  sp;                   /* stack pointer (next free slot) */
    CallFrame frames[SMOLLU_CALLSTACK_MAX];
    uint8_t  fp;                   /* call-frame pointer */

    /* Environment */
    Value     globals[256];
    NativeFn  natives[256];
} SmolluVM;

/* ────────────────────────────────────────────────────────────────────────── */
/*  Public API                                                               */
/* ────────────────────────────────────────────────────────────────────────── */

void smollu_vm_init(SmolluVM *vm);
void smollu_vm_prepare(SmolluVM *vm, const uint8_t *bytecode, const NativeFn *native_table);
void smollu_vm_load(SmolluVM *vm, const uint8_t *bytecode, size_t len);
void smollu_vm_register_native(SmolluVM *vm, uint8_t nat_id, NativeFn fn);
int  smollu_vm_run(SmolluVM *vm);
void smollu_vm_destroy(SmolluVM *vm);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SMOLLU_VM_H */