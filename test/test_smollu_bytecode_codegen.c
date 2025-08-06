/**
 * @file test_smollu_bytecode_codegen.c
 * @brief Unit tests for the Smollu bytecode code generator
 * 
 * Tests the compilation of AST nodes into executable bytecode, including:
 * - Basic expressions and literals
 * - Variable declarations and assignments
 * - Control flow statements (if/while)
 * - Function definitions and calls
 * - Native function calls
 * - Error handling and edge cases
 */

#include <criterion/criterion.h>
#include <stdint.h>
#include <string.h>

#include "../src/components/compiler/smollu_compiler.h"

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Bytecode opcode constants (duplicated for testing)                         */
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
/*  Test helper functions                                                       */
/* ──────────────────────────────────────────────────────────────────────────── */

static ASTNode *parse_code(const char *code) {
    Lexer lex;
    lexer_init(&lex, code);
    
    Parser p;
    parser_init(&p, &lex);
    
    ASTNode *root = parse_program(&p);
    cr_assert(root != NULL, "Failed to parse code");
    cr_assert(root->type == AST_PROGRAM, "Root is not a program node");
    
    return root;
}

static void verify_header(uint8_t *bytecode, size_t len, uint8_t device_id, uint8_t version, uint8_t func_count, uint8_t native_count) {
    cr_assert(len >= 16, "Bytecode too short for header");
    
    /* Magic "SMOL" */
    cr_assert_eq(bytecode[0], 'S', "Invalid magic byte 0");
    cr_assert_eq(bytecode[1], 'M', "Invalid magic byte 1");
    cr_assert_eq(bytecode[2], 'O', "Invalid magic byte 2");
    cr_assert_eq(bytecode[3], 'L', "Invalid magic byte 3");
    
    /* Version and device */
    cr_assert_eq(bytecode[4], version, "Version mismatch");
    cr_assert_eq(bytecode[5], device_id, "Device ID mismatch");
    
    /* Function and native counts */
    cr_assert_eq(bytecode[6], func_count, "Function count mismatch");
    cr_assert_eq(bytecode[7], native_count, "Native count mismatch");
    
    /* Code size (little-endian uint32 at offset 8) */
    uint32_t code_size = bytecode[8] | (bytecode[9] << 8) | (bytecode[10] << 16) | (bytecode[11] << 24);
    cr_assert_gt(code_size, 0, "Code size should be > 0");
    
    /* Reserved bytes should be 0 */
    cr_assert_eq(bytecode[12], 0, "Reserved byte 12 should be 0");
    cr_assert_eq(bytecode[13], 0, "Reserved byte 13 should be 0");
    cr_assert_eq(bytecode[14], 0, "Reserved byte 14 should be 0");
    cr_assert_eq(bytecode[15], 0, "Reserved byte 15 should be 0");
}

static uint8_t *get_code_section(uint8_t *bytecode, uint8_t native_count) {
    /* Code starts after 16-byte header + 2*native_count bytes for native table */
    return bytecode + 16 + (2 * native_count);
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Basic compilation tests                                                     */
/* ──────────────────────────────────────────────────────────────────────────── */

Test(bytecode_codegen, simple_program) {
    const char *code = 
        "init {}\n"
        "main { x = 42; }\n"
        "functions {}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    int result = smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    cr_assert_eq(result, 0, "Compilation failed");
    cr_assert(bytecode != NULL, "Bytecode is NULL");
    cr_assert_gt(len, 16, "Bytecode too short");
    
    /* Verify header */
    verify_header(bytecode, len, 0x00, 0x01, 0, 2); /* device 0 has 2 natives */
    
    /* Get code section */
    uint8_t *code_section = get_code_section(bytecode, 2);
    
    /* Should start with JMP to init section */
    cr_assert_eq(code_section[0], OP_JMP, "Expected JMP at code start");
    
    free(bytecode);
    ast_free(root);
}

Test(bytecode_codegen, literals) {
    const char *code = 
        "init {}\n"
        "main {\n"
        "  x = 42;\n"
        "  y = 3.14;\n"
        "  z = true;\n"
        "  w = false;\n"
        "  v = nil;\n"
        "}\n"
        "functions {}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    int result = smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    cr_assert_eq(result, 0, "Compilation failed");
    
    uint8_t *code_section = get_code_section(bytecode, 2);
    
    /* Skip initial JMP and find the literal pushes */
    /* This is a simplified check - in practice we'd need to parse the jump offset */
    bool found_push_i32 = false, found_push_f32 = false, found_push_true = false, found_push_false = false, found_push_nil = false;
    
    for (size_t i = 0; i < len - 16 - 4; i++) {
        if (code_section[i] == OP_PUSH_I32) { found_push_i32 = true;}
        if (code_section[i] == OP_PUSH_F32) { found_push_f32 = true;}
        if (code_section[i] == OP_PUSH_TRUE) { found_push_true = true;}
        if (code_section[i] == OP_PUSH_FALSE) { found_push_false = true;}
        if (code_section[i] == OP_PUSH_NIL) { found_push_nil = true;}
    }
    
    cr_assert(found_push_i32, "Should emit PUSH_I32 for integer literal");
    cr_assert(found_push_f32, "Should emit PUSH_F32 for float literal");
    cr_assert(found_push_true, "Should emit PUSH_TRUE for true literal");
    cr_assert(found_push_false, "Should emit PUSH_FALSE for false literal");
    cr_assert(found_push_nil, "Should emit PUSH_NIL for nil literal");
    
    free(bytecode);
    ast_free(root);
}

Test(bytecode_codegen, arithmetic_expressions) {
    const char *code = 
        "init {}\n"
        "main {\n"
        "  x = 1 + 2;\n"
        "  y = 5 - 3;\n"
        "  z = 4 * 6;\n"
        "  w = 8 / 2;\n"
        "  v = 10 % 3;\n"
        "  u = -5;\n"
        "  t = -u;\n"
        "  a = true && false;\n"
        "  b = true || false;\n"
        "}\n"
        "functions {}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    int result = smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    cr_assert_eq(result, 0, "Compilation failed");
    
    uint8_t *code_section = get_code_section(bytecode, 2);
    
    /* Check for arithmetic opcodes */
    bool found_add = false, found_sub = false, found_mul = false, found_div = false, found_mod = false, found_neg = false, found_and = false, found_or = false;
    
    for (size_t i = 0; i < len - 16 - 4; i++) {
        if (code_section[i] == OP_ADD) found_add = true;
        if (code_section[i] == OP_SUB) found_sub = true;
        if (code_section[i] == OP_MUL) found_mul = true;
        if (code_section[i] == OP_DIV) found_div = true;
        if (code_section[i] == OP_MOD) found_mod = true;
        if (code_section[i] == OP_NEG) found_neg = true;
        if (code_section[i] == OP_AND) found_and = true;
        if (code_section[i] == OP_OR) found_or = true;
    }
    
    cr_assert(found_add, "Should emit ADD for + operator");
    cr_assert(found_sub, "Should emit SUB for - operator");
    cr_assert(found_mul, "Should emit MUL for * operator");
    cr_assert(found_div, "Should emit DIV for / operator");
    cr_assert(found_mod, "Should emit MOD for %% operator");
    cr_assert(found_neg, "Should emit NEG for unary - operator");
    cr_assert(found_and, "Should emit AND for && operator");
    cr_assert(found_or, "Should emit OR for || operator");
    
    free(bytecode);
    ast_free(root);
}

Test(bytecode_codegen, comparison_expressions) {
    const char *code = 
        "init {}\n"
        "main {\n"
        "  1 == 2;\n"
        "  3 != 4;\n"
        "  5 < 6;\n"
        "  7 <= 8;\n"
        "  9 > 10;\n"
        "  11 >= 12;\n"
        "}\n"
        "functions {}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    int result = smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    cr_assert_eq(result, 0, "Compilation failed");
    
    uint8_t *code_section = get_code_section(bytecode, 2);
    
    /* Check for comparison opcodes */
    bool found_eq = false, found_neq = false, found_lt = false, found_le = false, found_gt = false, found_ge = false;
    
    for (size_t i = 0; i < len - 16 - 4; i++) {
        if (code_section[i] == OP_EQ) found_eq = true;
        if (code_section[i] == OP_NEQ) found_neq = true;
        if (code_section[i] == OP_LT) found_lt = true;
        if (code_section[i] == OP_LE) found_le = true;
        if (code_section[i] == OP_GT) found_gt = true;
        if (code_section[i] == OP_GE) found_ge = true;
    }
    
    cr_assert(found_eq, "Should emit EQ for == operator");
    cr_assert(found_neq, "Should emit NEQ for != operator");
    cr_assert(found_lt, "Should emit LT for < operator");
    cr_assert(found_le, "Should emit LE for <= operator");
    cr_assert(found_gt, "Should emit GT for > operator");
    cr_assert(found_ge, "Should emit GE for >= operator");
    
    free(bytecode);
    ast_free(root);
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Variable and assignment tests                                              */
/* ──────────────────────────────────────────────────────────────────────────── */

Test(bytecode_codegen, global_variables) {
    const char *code = 
        "init { x = 42; }\n"
        "main { y = x + 1; }\n"
        "functions {}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    int result = smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    cr_assert_eq(result, 0, "Compilation failed");
    
    uint8_t *code_section = get_code_section(bytecode, 2);
    
    /* Check for global load/store operations */
    bool found_store_global = false, found_load_global = false;
    
    for (size_t i = 0; i < len - 16 - 4; i++) {
        if (code_section[i] == OP_STORE_GLOBAL) found_store_global = true;
        if (code_section[i] == OP_LOAD_GLOBAL) found_load_global = true;
    }
    
    cr_assert(found_store_global, "Should emit STORE_GLOBAL for global assignment");
    cr_assert(found_load_global, "Should emit LOAD_GLOBAL for global variable access");
    
    free(bytecode);
    ast_free(root);
}

Test(bytecode_codegen, local_variables) {
    const char *code = 
        "init {}\n"
        "main {}\n"
        "functions {\n"
        "  function test() {\n"
        "    local x = 42;\n"
        "    local y = x + 1;\n"
        "    return y;\n"
        "  }\n"
        "}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    int result = smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    cr_assert_eq(result, 0, "Compilation failed");
    
    /* Verify function count in header */
    verify_header(bytecode, len, 0x00, 0x01, 1, 2);
    
    uint8_t *code_section = get_code_section(bytecode, 2);
    
    /* Check for local load/store operations */
    bool found_store_local = false, found_load_local = false;
    
    for (size_t i = 0; i < len - 16 - 4; i++) {
        if (code_section[i] == OP_STORE_LOCAL) found_store_local = true;
        if (code_section[i] == OP_LOAD_LOCAL) found_load_local = true;
    }
    
    cr_assert(found_store_local, "Should emit STORE_LOCAL for local assignment");
    cr_assert(found_load_local, "Should emit LOAD_LOCAL for local variable access");
    
    free(bytecode);
    ast_free(root);
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Control flow tests                                                         */
/* ──────────────────────────────────────────────────────────────────────────── */

Test(bytecode_codegen, if_statement) {
    const char *code = 
        "init {}\n"
        "main {\n"
        "  if (true) {\n"
        "    x = 1;\n"
        "  }\n"
        "}\n"
        "functions {}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    int result = smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    cr_assert_eq(result, 0, "Compilation failed");
    
    uint8_t *code_section = get_code_section(bytecode, 2);
    
    /* Check for conditional jump */
    bool found_jmp_if_false = false;
    
    for (size_t i = 0; i < len - 16 - 4; i++) {
        if (code_section[i] == OP_JMP_IF_FALSE) found_jmp_if_false = true;
    }
    
    cr_assert(found_jmp_if_false, "Should emit JMP_IF_FALSE for if statement");
    
    free(bytecode);
    ast_free(root);
}

Test(bytecode_codegen, if_else_statement) {
    const char *code = 
        "init {}\n"
        "main {\n"
        "  if (true) {\n"
        "    x = 1;\n"
        "  } else {\n"
        "    x = 2;\n"
        "  }\n"
        "}\n"
        "functions {}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    int result = smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    cr_assert_eq(result, 0, "Compilation failed");
    
    uint8_t *code_section = get_code_section(bytecode, 2);
    
    /* Check for conditional and unconditional jumps */
    bool found_jmp_if_false = false, found_jmp = false;
    
    for (size_t i = 0; i < len - 16 - 4; i++) {
        if (code_section[i] == OP_JMP_IF_FALSE) found_jmp_if_false = true;
        if (code_section[i] == OP_JMP) found_jmp = true;
    }
    
    cr_assert(found_jmp_if_false, "Should emit JMP_IF_FALSE for if condition");
    cr_assert(found_jmp, "Should emit JMP to skip else branch");
    
    free(bytecode);
    ast_free(root);
}

Test(bytecode_codegen, while_loop) {
    const char *code = 
        "init { x = 0; }\n"
        "main {\n"
        "  while (x < 10) {\n"
        "    x = x + 1;\n"
        "  }\n"
        "}\n"
        "functions {}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    int result = smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    cr_assert_eq(result, 0, "Compilation failed");
    
    uint8_t *code_section = get_code_section(bytecode, 2);
    
    /* Check for loop structure: JMP_IF_FALSE for exit, JMP for back-branch */
    bool found_jmp_if_false = false, found_jmp = false;
    
    for (size_t i = 0; i < len - 16 - 4; i++) {
        if (code_section[i] == OP_JMP_IF_FALSE) found_jmp_if_false = true;
        if (code_section[i] == OP_JMP) found_jmp = true;
    }
    
    cr_assert(found_jmp_if_false, "Should emit JMP_IF_FALSE for loop exit");
    cr_assert(found_jmp, "Should emit JMP for loop back-branch");
    
    free(bytecode);
    ast_free(root);
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Function definition and call tests                                         */
/* ──────────────────────────────────────────────────────────────────────────── */

Test(bytecode_codegen, function_definition) {
    const char *code = 
        "init {}\n"
        "main {}\n"
        "functions {\n"
        "  function add(a, b) {\n"
        "    return a + b;\n"
        "  }\n"
        "}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    int result = smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    cr_assert_eq(result, 0, "Compilation failed");
    
    /* Verify function count in header */
    verify_header(bytecode, len, 0x00, 0x01, 1, 2);
    
    uint8_t *code_section = get_code_section(bytecode, 2);
    
    /* Check for return instruction */
    bool found_ret = false;
    
    for (size_t i = 0; i < len - 16 - 4; i++) {
        if (code_section[i] == OP_RET) found_ret = true;
    }
    
    cr_assert(found_ret, "Should emit RET for function return");
    
    free(bytecode);
    ast_free(root);
}

Test(bytecode_codegen, function_definition_with_no_return) {
    const char *code = 
        "init {}\n"
        "main {}\n"
        "functions {\n"
        "  function test() {}\n"
        "}";

    ASTNode *root = parse_code(code);

    uint8_t *bytecode;
    size_t len;
    int result = smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    cr_assert_eq(result, 0, "Compilation failed");
    
    uint8_t *code_section = get_code_section(bytecode, 2);
    
    /* Check for return instruction */
    bool found_ret = false;
    
    for (size_t i = 0; i < len - 16 - 4; i++) {
        if (code_section[i] == OP_RET) found_ret = true;
    }
    
    cr_assert(found_ret, "Should emit RET for function return");
    
    free(bytecode);
    ast_free(root);
}


Test(bytecode_codegen, function_call) {
    const char *code = 
        "init {}\n"
        "main {\n"
        "  add(1, 2);\n"
        "}\n"
        "functions {\n"
        "  function add(a, b) {\n"
        "    return a + b;\n"
        "  }\n"
        "}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    int result = smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    cr_assert_eq(result, 0, "Compilation failed");
    
    uint8_t *code_section = get_code_section(bytecode, 2);
    
    /* Check for function call instruction */
    bool found_call = false;
    
    for (size_t i = 0; i < len - 16 - 4; i++) {
        if (code_section[i] == OP_CALL) found_call = true;
    }
    
    cr_assert(found_call, "Should emit CALL for function call");
    
    free(bytecode);
    ast_free(root);
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Native function call tests                                                 */
/* ──────────────────────────────────────────────────────────────────────────── */

Test(bytecode_codegen, native_call) {
    const char *code = 
        "init {\n"
        "  native print(42);\n"
        "}\n"
        "main {}\n"
        "functions {}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    int result = smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    cr_assert_eq(result, 0, "Compilation failed");
    
    uint8_t *code_section = get_code_section(bytecode, 2);
    
    /* Check for native call instruction */
    bool found_ncall = false;
    
    for (size_t i = 0; i < len - 16 - 4; i++) {
        if (code_section[i] == OP_NCALL) found_ncall = true;
    }
    
    cr_assert(found_ncall, "Should emit NCALL for native function call");
    
    free(bytecode);
    ast_free(root);
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Error handling tests                                                       */
/* ──────────────────────────────────────────────────────────────────────────── */

Test(bytecode_codegen, undeclared_variable_error, .exit_code = 1) {
    const char *code = 
        "init {}\n"
        "main {\n"
        "  x = undeclared_var;\n"
        "}\n"
        "functions {}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    
    /* This should fail with an error about undeclared variable */
    smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    ast_free(root);
}

Test(bytecode_codegen, unknown_native_error, .exit_code = 1) {
    const char *code = 
        "init {\n"
        "  native unknown_function();\n"
        "}\n"
        "main {}\n"
        "functions {}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    
    /* This should fail with an error about unknown native function */
    smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    ast_free(root);
}

Test(bytecode_codegen, invalid_device_error, .exit_code = 1) {
    const char *code = 
        "init {}\n"
        "main {}\n"
        "functions {}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    
    /* This should fail with an error about unknown device ID */
    smollu_generate_bytecode(root, 0xFF, 0x01, &bytecode, &len);
    
    ast_free(root);
}

Test(bytecode_codegen, null_root_error, .exit_code = 1) {
    uint8_t *bytecode;
    size_t len;
    
    /* This should fail with an error about NULL root */
    smollu_generate_bytecode(NULL, 0x00, 0x01, &bytecode, &len);
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Complex integration tests                                                  */
/* ──────────────────────────────────────────────────────────────────────────── */

Test(bytecode_codegen, complex_program) {
    const char *code = 
        "init {\n"
        "  counter = 0;\n"
        "  native print(counter);\n"
        "}\n"
        "main {\n"
        "  while (counter < 10) {\n"
        "    if (counter % 2 == 0) {\n"
        "      native print(counter);\n"
        "    } else {\n"
        "      counter = counter + 1;\n"
        "    }\n"
        "    counter = increment(counter);\n"
        "  }\n"
        "}\n"
        "functions {\n"
        "  function increment(x) {\n"
        "    local result = x + 1;\n"
        "    return result;\n"
        "  }\n"
        "}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    int result = smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    cr_assert_eq(result, 0, "Complex program compilation failed");
    
    /* Verify header with one function */
    verify_header(bytecode, len, 0x00, 0x01, 1, 2);
    
    uint8_t *code_section = get_code_section(bytecode, 2);
    
    /* Check that we have all expected opcodes */
    bool found_opcodes[256] = {false};
    
    for (size_t i = 0; i < len - 16 - 4; i++) {
        found_opcodes[code_section[i]] = true;
    }
    
    cr_assert(found_opcodes[OP_PUSH_I32], "Should have integer pushes");
    cr_assert(found_opcodes[OP_STORE_GLOBAL], "Should have global stores");
    cr_assert(found_opcodes[OP_LOAD_GLOBAL], "Should have global loads");
    cr_assert(found_opcodes[OP_STORE_LOCAL], "Should have local stores");
    cr_assert(found_opcodes[OP_LOAD_LOCAL], "Should have local loads");
    cr_assert(found_opcodes[OP_ADD], "Should have addition");
    cr_assert(found_opcodes[OP_MOD], "Should have modulo");
    cr_assert(found_opcodes[OP_EQ], "Should have equality");
    cr_assert(found_opcodes[OP_LT], "Should have less than");
    cr_assert(found_opcodes[OP_JMP_IF_FALSE], "Should have conditional jumps");
    cr_assert(found_opcodes[OP_JMP], "Should have unconditional jumps");
    cr_assert(found_opcodes[OP_CALL], "Should have function calls");
    cr_assert(found_opcodes[OP_RET], "Should have returns");
    cr_assert(found_opcodes[OP_NCALL], "Should have native calls");
    cr_assert(found_opcodes[OP_HALT], "Should have halt at end");
    
    free(bytecode);
    ast_free(root);
}

Test(bytecode_codegen, nested_control_flow) {
    const char *code = 
        "init {}\n"
        "main {\n"
        "  x = 0;\n"
        "  while (x < 5) {\n"
        "    if (x < 3) {\n"
        "      while (x < 2) {\n"
        "        x = x + 1;\n"
        "      }\n"
        "    } else {\n"
        "      x = x + 2;\n"
        "    }\n"
        "    x = x + 1;\n"
        "  }\n"
        "}\n"
        "functions {}";
    
    ASTNode *root = parse_code(code);
    
    uint8_t *bytecode;
    size_t len;
    int result = smollu_generate_bytecode(root, 0x00, 0x01, &bytecode, &len);
    
    cr_assert_eq(result, 0, "Nested control flow compilation failed");
    
    /* Count jumps to ensure proper nesting */
    uint8_t *code_section = get_code_section(bytecode, 2);
    int jmp_count = 0, jmp_if_false_count = 0;
    
    for (size_t i = 0; i < len - 16 - 4; i++) {
        if (code_section[i] == OP_JMP) jmp_count++;
        if (code_section[i] == OP_JMP_IF_FALSE) jmp_if_false_count++;
    }
    
    cr_assert_gt(jmp_count, 2, "Should have multiple unconditional jumps for nested structure");
    cr_assert_gt(jmp_if_false_count, 2, "Should have multiple conditional jumps for nested structure");
    
    free(bytecode);
    ast_free(root);
}