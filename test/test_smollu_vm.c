#include <criterion/criterion.h>
#include <stdint.h>
#include "../src/components/vm/smollu_vm.h"

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
};

/* ────────────────────────────────────────────────────────────────────────── */
/*  Helper utilities                                                          */
/* ────────────────────────────────────────────────────────────────────────── */

static SmolluVM create_and_run(const uint8_t *code, size_t len) {
    SmolluVM vm; smollu_vm_init(&vm);
    smollu_vm_load(&vm, code, len);
    int rc = smollu_vm_run(&vm);
    cr_assert_eq(rc, 0, "VM returned error code %d", rc);
    return vm; /* NOTE: copy of struct is fine (POD) */
}

static void expect_int(const SmolluVM *vm, int32_t expected) {
    cr_assert_eq(vm->sp, 1, "Expected stack size 1, got %u", vm->sp);
    cr_assert_eq(vm->stack[0].type, VAL_INT, "Top of stack not INT");
    cr_assert_eq(vm->stack[0].as.i, expected, "Expected %d on stack, got %d", expected, vm->stack[0].as.i);
}

static void expect_bool(const SmolluVM *vm, bool expected) {
    cr_assert_eq(vm->sp, 1, "Expected stack size 1, got %u", vm->sp);
    cr_assert_eq(vm->stack[0].type, VAL_BOOL, "Top of stack not BOOL");
    cr_assert_eq(vm->stack[0].as.boolean, expected, "Expected %d on stack, got %d", expected, vm->stack[0].as.boolean);
}

static void expect_float(const SmolluVM *vm, float expected) {
    cr_assert_eq(vm->sp, 1, "Expected stack size 1, got %u", vm->sp);
    cr_assert_eq(vm->stack[0].type, VAL_FLOAT, "Top of stack not FLOAT");
    cr_assert_eq(vm->stack[0].as.f, expected, "Expected %f on stack, got %f", expected, vm->stack[0].as.f);
}

/* ────────────────────────────────────────────────────────────────────────── */
/*  Tests                                                                     */
/* ────────────────────────────────────────────────────────────────────────── */

/* ────────────────────────────────────────────────────────────────────────── */
/*  Arithmetic tests                                                          */
/* ────────────────────────────────────────────────────────────────────────── */
Test(vm, arithmetic_add_i8) {
    const uint8_t code[] = {
        OP_PUSH_I8, 5,
        OP_PUSH_I8, 3,
        OP_ADD,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_int(&vm, 8);

}

Test(vm, arithmetic_sub_i8) {
    const uint8_t code[] = {
        OP_PUSH_I8, 3,
        OP_PUSH_I8, 5,
        OP_SUB,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_int(&vm, -2);
}

Test(vm, arithmetic_mul_i8) {
    const uint8_t code[] = {
        OP_PUSH_I8, 3,
        OP_PUSH_I8, 5,
        OP_MUL,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_int(&vm, 15);
}

Test(vm, arithmetic_div_i8) {
    const uint8_t code[] = {
        OP_PUSH_I8, 10,
        OP_PUSH_I8, 2,
        OP_DIV,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_int(&vm, 5);
}

Test(vm, arithmetic_mod_i8) {
    const uint8_t code[] = {
        OP_PUSH_I8, 10,
        OP_PUSH_I8, 3,
        OP_MOD,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_int(&vm, 1);
}

Test(vm, neg_and_not_i8) {
    /* Program 1: NEG */
    const uint8_t prog_neg[] = { OP_PUSH_I8, 7, OP_NEG, OP_HALT };
    SmolluVM vm_neg = create_and_run(prog_neg, sizeof(prog_neg));
    expect_int(&vm_neg, -7);

    /* Program 2: NOT true */
    const uint8_t prog_not[] = { OP_PUSH_TRUE, OP_NOT, OP_HALT };
    SmolluVM vm_not = create_and_run(prog_not, sizeof(prog_not));
    expect_bool(&vm_not, false);
}

Test(vm, comparison_eq_i8) {
    const uint8_t code[] = {
        OP_PUSH_I8, 1,
        OP_PUSH_I8, 1,
        OP_EQ,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_bool(&vm, true);
}

Test(vm, arithmetic_add_i32) {
    const uint8_t code[] = {
        OP_PUSH_I32, 5, 0, 0, 0,
        OP_PUSH_I32, 3, 0, 0, 0,
        OP_ADD,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_int(&vm, 8);
}

Test(vm, arithmetic_sub_i32) {
    const uint8_t code[] = {
        OP_PUSH_I32, 3, 0, 0, 0,
        OP_PUSH_I32, 5, 0, 0, 0,
        OP_SUB,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_int(&vm, -2);
}

Test(vm, arithmetic_mul_i32) {
    const uint8_t code[] = {
        OP_PUSH_I32, 3, 0, 0, 0,
        OP_PUSH_I32, 5, 0, 0, 0,
        OP_MUL,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_int(&vm, 15);
}

Test(vm, arithmetic_div_i32) {
    const uint8_t code[] = {
        OP_PUSH_I32, 10, 0, 0, 0,
        OP_PUSH_I32, 2, 0, 0, 0,
        OP_DIV,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_int(&vm, 5);
}

Test(vm, arithmetic_mod_i32) {
    const uint8_t code[] = {
        OP_PUSH_I32, 10, 0, 0, 0,
        OP_PUSH_I32, 3, 0, 0, 0,
        OP_MOD,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_int(&vm, 1);
}

Test(vm, neg_and_not_i32) {
    const uint8_t prog_neg[] = { OP_PUSH_I32, 7, 0, 0, 0, OP_NEG, OP_HALT };
    SmolluVM vm_neg = create_and_run(prog_neg, sizeof(prog_neg));
    expect_int(&vm_neg, -7);
}

Test(vm, comparison_eq_i32) {
    const uint8_t code[] = {
        OP_PUSH_I32, 1, 0, 0, 0,
        OP_PUSH_I32, 1, 0, 0, 0,
        OP_EQ,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_bool(&vm, true);
}

Test(vm, arithmetic_add_f32) {
    const uint8_t code[] = {
        OP_PUSH_F32, 0x00, 0x00, 0xa0, 0x40,
        OP_PUSH_F32, 0x00, 0x00, 0x40, 0x40,
        OP_ADD,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_float(&vm, 8.0f);
}

Test(vm, arithmetic_sub_f32) {
    const uint8_t code[] = {
        OP_PUSH_F32, 0x00, 0x00, 0x40, 0x40,
        OP_PUSH_F32, 0x00, 0x00, 0xa0, 0x40,
        OP_SUB,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_float(&vm, -2.0f);
}

Test(vm, arithmetic_mul_f32) {
    const uint8_t code[] = {
        OP_PUSH_F32, 0x00, 0x00, 0x40, 0x40,
        OP_PUSH_F32, 0x00, 0x00, 0xa0, 0x40,
        OP_MUL,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_float(&vm, 15.0f);
}

Test(vm, arithmetic_div_f32) {
    const uint8_t code[] = {
        OP_PUSH_F32, 0x00, 0x00, 0x40, 0x40,
        OP_PUSH_F32, 0x00, 0x00, 0xa0, 0x40,
        OP_DIV,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_float(&vm, 0.6f);
}

Test(vm, comparison_eq_f32) {
    const uint8_t code[] = {
        OP_PUSH_F32, 0x00, 0x00, 0x40, 0x40,
        OP_PUSH_F32, 0x00, 0x00, 0x40, 0x40,
        OP_EQ,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_bool(&vm, true);
}

Test(vm, comparison_neq_f32) {
    const uint8_t code[] = {
        OP_PUSH_F32, 0x00, 0x00, 0x40, 0x40,
        OP_PUSH_F32, 0x00, 0x00, 0x40, 0x41,
        OP_NEQ,
        OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_bool(&vm, true);
}


/* ────────────────────────────────────────────────────────────────────────── */
/*  Globals & locals tests                                                    */
/* ────────────────────────────────────────────────────────────────────────── */
Test(vm, globals_and_locals_i8) {
    /* Globals */
    const uint8_t prog_global[] = {
        OP_PUSH_I8, 42,
        OP_STORE_GLOBAL, 5,
        OP_LOAD_GLOBAL, 5,
        OP_HALT
    };
    SmolluVM vm_g = create_and_run(prog_global, sizeof(prog_global));
    expect_int(&vm_g, 42);

    /* Locals (slot 0) */
    const uint8_t prog_local[] = {
        OP_PUSH_I8, 7,
        OP_STORE_LOCAL, 0,
        OP_LOAD_LOCAL, 0,
        OP_HALT
    };
    SmolluVM vm_l = create_and_run(prog_local, sizeof(prog_local));
    expect_int(&vm_l, 7);
}

Test(vm, globals_and_locals_i32) {
    /* Globals */
    const uint8_t prog_global[] = {
        OP_PUSH_I32, 42, 0, 0, 0,
        OP_STORE_GLOBAL, 5,
        OP_LOAD_GLOBAL, 5,
        OP_HALT
    };
    SmolluVM vm_g = create_and_run(prog_global, sizeof(prog_global));
    expect_int(&vm_g, 42);

    /* Locals (slot 0) */
    const uint8_t prog_local[] = {
        OP_PUSH_I32, 7, 0, 0, 0,
        OP_STORE_LOCAL, 0,
        OP_LOAD_LOCAL, 0,
        OP_HALT
    };
    SmolluVM vm_l = create_and_run(prog_local, sizeof(prog_local));
    expect_int(&vm_l, 7);
}

Test(vm, globals_and_locals_f32) {
    /* Globals */
    const uint8_t prog_global[] = {
        OP_PUSH_F32, 0x00, 0x00, 0x40, 0x40,
        OP_STORE_GLOBAL, 5,
        OP_LOAD_GLOBAL, 5,
        OP_HALT
    };
    SmolluVM vm_g = create_and_run(prog_global, sizeof(prog_global));
    expect_float(&vm_g, 3.0f);

    /* Locals (slot 0) */
    const uint8_t prog_local[] = {
        OP_PUSH_F32, 0x00, 0x00, 0x40, 0x40,
        OP_STORE_LOCAL, 0,
        OP_LOAD_LOCAL, 0,
        OP_HALT
    };
    SmolluVM vm_l = create_and_run(prog_local, sizeof(prog_local));
    expect_float(&vm_l, 3.0f);
}

/* ────────────────────────────────────────────────────────────────────────── */
/*  Native Ccall and Function tests                                           */
/* ────────────────────────────────────────────────────────────────────────── */
/* Native helper */
static Value native_sum(Value *args, uint8_t argc) {
    int32_t sum = 0;
    for (uint8_t i = 0; i < argc; ++i) {
        sum += args[i].as.i;
    }
    return value_from_int(sum);
}

Test(vm, native_call) {
    const uint8_t code[] = {
        OP_PUSH_I8, 2,
        OP_PUSH_I8, 3,
        OP_NCALL, 0, 2, /* nat_id=0, argc=2 */
        OP_HALT
    };
    SmolluVM vm; smollu_vm_init(&vm);
    smollu_vm_register_native(&vm, 0, native_sum);
    smollu_vm_load(&vm, code, sizeof(code));
    int rc = smollu_vm_run(&vm);
    cr_assert_eq(rc, 0);
    expect_int(&vm, 5);
}

Test(vm, call_and_ret) {
    const uint8_t code[] = {
        /* 0 */ OP_PUSH_I8, 10,
        /* 2 */ OP_CALL, 0x07, 0x00, 1, /* addr=6, argc=1 */
        /* 5 */ OP_HALT,
        /* 6 */ OP_LOAD_LOCAL, 0,
        /* 8 */ OP_PUSH_I8, 1,
        /*10 */ OP_ADD,
        /*11 */ OP_RET, 1
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_int(&vm, 11);
} 

/* ────────────────────────────────────────────────────────────────────────── */
/*  Control flow tests                                                        */
/* ────────────────────────────────────────────────────────────────────────── */
Test(vm, jmp) {
    const uint8_t code[] = {
        /* 0 */ OP_JMP, 10, 0,
        /* 3 */ OP_NOP,
        /* 4 */ OP_HALT,
        /* 5 */ OP_NOP,
        /* 6 */ OP_NOP,
        /* 7 */ OP_NOP,
        /* 8 */ OP_NOP,
        /* 9 */ OP_NOP,
        /*10 */ OP_NOP,
        /*11 */ OP_NOP,
        /*12 */ OP_NOP,
        /*13 */ OP_PUSH_I8, 10,
        /*14 */ OP_HALT
    };
    SmolluVM vm = create_and_run(code, sizeof(code));
    expect_int(&vm, 10);
}

/* ────────────────────────────────────────────────────────────────────────── */
/*  Prepare / Header / Native table tests                                     */
/* ────────────────────────────────────────────────────────────────────────── */
Test(vm, prepare_and_native_table) {
    /* Device-wide native function table */
    static const NativeFn device_table[] = { native_sum };

    /* Header (16B) + native table (2B) + code (8B) = 26 bytes */
    static const uint8_t prog[] = {
        /* Header */
        'S','M','O','L',       /* magic */
        0x01,                  /* version */
        0x00,                  /* device_id */
        0x01,                  /* native_count */
        0x08,0x00,0x00,0x00,   /* code_size */
        0x00,0x00,0x00,0x00,   /* reserved */
        0x00,                  /* function_count (unused/padding) */
        /* Native table (1 entry -> index 0) */
        0x00,0x00,
        /* Code section */
        OP_PUSH_I8, 2,
        OP_PUSH_I8, 3,
        OP_NCALL, 0, 2,        /* nat_id=0, argc=2 */
        OP_HALT
    };

    const size_t HEADER_SIZE = 16;
    const size_t TABLE_SIZE  = 2;
    const size_t CODE_OFFSET = HEADER_SIZE + TABLE_SIZE;

    SmolluVM vm; smollu_vm_init(&vm);

    /* Inject header + native mapping */
    smollu_vm_prepare(&vm, prog, device_table);

    /* Load and execute the code portion */
    smollu_vm_load(&vm, prog + CODE_OFFSET, sizeof(prog) - CODE_OFFSET);
    int rc = smollu_vm_run(&vm);
    cr_assert_eq(rc, 0);

    /* Validate that the native table was properly registered */
    cr_assert_eq(vm.native_count, 1);
    cr_assert(vm.natives[0] == native_sum, "Native table registration failed");

    /* Ensure native execution returned expected result */
    expect_int(&vm, 5);
}
