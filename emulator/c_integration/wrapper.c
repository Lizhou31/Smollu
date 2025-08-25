/**
 * @file wrapper.c
 * @brief Implementation of C wrapper functions for Smollu VM FFI
 */

#include "wrapper.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Static state for emulator features                                          */
/* ──────────────────────────────────────────────────────────────────────────── */

static char print_output_buffer[4096] = {0};
static size_t print_output_length = 0;
static char all_print_output[8192] = {0};
static size_t all_output_length = 0;
static OutputCallback output_callback = NULL;
static CompletionCallback completion_callback = NULL;
static volatile int pending_native_calls = 0;
static volatile int vm_exit_code = 0;
static volatile int vm_has_finished = 0;

/* ──────────────────────────────────────────────────────────────────────────── */
/*  VM Lifecycle Management                                                     */
/* ──────────────────────────────────────────────────────────────────────────── */

SmolluVM* wrapper_vm_create(void) {
    SmolluVM* vm = (SmolluVM*)malloc(sizeof(SmolluVM));
    if (!vm) {
        return NULL;
    }

    smollu_vm_init(vm);
    return vm;
}

void wrapper_vm_destroy(SmolluVM* vm) {
    if (vm) {
        smollu_vm_destroy(vm);
        free(vm);
    }
}

int wrapper_vm_load_bytecode(SmolluVM* vm, const uint8_t* bytecode, size_t length) {
    if (!vm || !bytecode || length < 16) {
        return -1;
    }

    // Parse header to get native function count
    uint8_t native_count = bytecode[7];

    // For now, we'll use the emulator device (ID 0x01) and register our own natives
    // Calculate code section offset
    size_t code_offset = 16 + (size_t)native_count * 2;
    if (length < code_offset) {
        return -2; // Invalid bytecode
    }

    // Prepare VM with header
    smollu_vm_prepare(vm, bytecode, NULL); // We'll register natives separately

    // Load code section
    const uint8_t* code_ptr = bytecode + code_offset;
    size_t code_len = length - code_offset;
    smollu_vm_load(vm, code_ptr, code_len);

    // Register emulator-specific native functions
    wrapper_vm_register_native(vm, 0, emulator_native_print);
    wrapper_vm_register_native(vm, 1, emulator_native_led_matrix_init);
    wrapper_vm_register_native(vm, 2, emulator_native_led_set);
    wrapper_vm_register_native(vm, 3, emulator_native_led_set_color);
    wrapper_vm_register_native(vm, 4, emulator_native_led_clear);
    wrapper_vm_register_native(vm, 5, emulator_native_led_set_row);
    wrapper_vm_register_native(vm, 6, emulator_native_led_set_col);
    wrapper_vm_register_native(vm, 7, emulator_native_led_get);
    wrapper_vm_register_native(vm, 8, emulator_native_delay_ms);

    return 0;
}

int wrapper_vm_run(SmolluVM* vm) {
    if (!vm) {
        return -1;
    }

    // Reset state for new execution
    pending_native_calls = 0;
    vm_has_finished = 0;

    int result = smollu_vm_run(vm);

    // Store the exit code and mark VM as finished
    vm_exit_code = result;
    vm_has_finished = 1;

    // If we have a completion callback and no pending native calls, call it immediately
    // Otherwise, the last native function will call it when it completes
    if (completion_callback && pending_native_calls == 0) {
        completion_callback(result);
    }

    return result;
}

void wrapper_vm_reset(SmolluVM* vm) {
    if (!vm) {
        return;
    }

    // Reset execution state but keep bytecode loaded
    vm->pc = 0;
    vm->sp = 0;
    vm->fp = 0;

    // Clear stack
    memset(vm->stack, 0, sizeof(vm->stack));

    // Clear call frames
    memset(vm->frames, 0, sizeof(vm->frames));

    // Keep globals and natives as they are
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Native Function Management                                                  */
/* ──────────────────────────────────────────────────────────────────────────── */

int wrapper_vm_register_native(SmolluVM* vm, uint8_t function_id, NativeFn function) {
    if (!vm || !function) {
        return -1;
    }

    smollu_vm_register_native(vm, function_id, function);
    return 0;
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  VM State Inspection                                                         */
/* ──────────────────────────────────────────────────────────────────────────── */

size_t wrapper_vm_get_pc(const SmolluVM* vm) {
    return vm ? vm->pc : 0;
}

uint8_t wrapper_vm_get_sp(const SmolluVM* vm) {
    return vm ? vm->sp : 0;
}

Value wrapper_vm_get_stack_value(const SmolluVM* vm, uint8_t offset) {
    if (!vm || offset >= vm->sp) {
        return value_make_nil();
    }

    // Stack grows upward, so top is at sp-1
    return vm->stack[vm->sp - 1 - offset];
}

Value wrapper_vm_get_global(const SmolluVM* vm, uint8_t slot) {
    if (!vm) {
        return value_make_nil();
    }

    return vm->globals[slot];
}

void wrapper_vm_set_global(SmolluVM* vm, uint8_t slot, Value value) {
    if (vm) {
        vm->globals[slot] = value;
    }
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Value Creation Helpers                                                      */
/* ──────────────────────────────────────────────────────────────────────────── */

Value wrapper_value_nil(void) {
    return value_make_nil();
}

Value wrapper_value_bool(bool b) {
    return value_from_bool(b);
}

Value wrapper_value_int(int32_t i) {
    return value_from_int(i);
}

Value wrapper_value_float(float f) {
    return value_from_float(f);
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Value Type Checking and Extraction                                          */
/* ──────────────────────────────────────────────────────────────────────────── */

ValueType wrapper_value_type(Value val) {
    return val.type;
}

bool wrapper_value_is_nil(Value val) {
    return val.type == VAL_NIL;
}

bool wrapper_value_is_bool(Value val) {
    return val.type == VAL_BOOL;
}

bool wrapper_value_is_int(Value val) {
    return val.type == VAL_INT;
}

bool wrapper_value_is_float(Value val) {
    return val.type == VAL_FLOAT;
}

bool wrapper_value_as_bool(Value val) {
    return val.as.boolean;
}

int32_t wrapper_value_as_int(Value val) {
    return val.as.i;
}

float wrapper_value_as_float(Value val) {
    return val.as.f;
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Emulator-specific Native Functions                                          */
/* ──────────────────────────────────────────────────────────────────────────── */

Value emulator_native_print(Value* args, uint8_t argc) {
    // Increment pending native call counter
    __sync_fetch_and_add(&pending_native_calls, 1);

    // Clear current output line buffer
    print_output_length = 0;
    print_output_buffer[0] = '\0';

    // Format output into current line buffer
    for (uint8_t i = 0; i < argc; ++i) {
        Value v = args[i];
        char temp[64];
        size_t temp_len = 0;

        switch (v.type) {
            case VAL_NIL:
                strcpy(temp, "nil");
                temp_len = 3;
                break;
            case VAL_BOOL:
                if (v.as.boolean) {
                    strcpy(temp, "true");
                    temp_len = 4;
                } else {
                    strcpy(temp, "false");
                    temp_len = 5;
                }
                break;
            case VAL_INT:
                temp_len = snprintf(temp, sizeof(temp), "%d", v.as.i);
                break;
            case VAL_FLOAT:
                temp_len = snprintf(temp, sizeof(temp), "%f", v.as.f);
                break;
        }

        // Append to current line buffer if there's space
        if (print_output_length + temp_len + 1 < sizeof(print_output_buffer)) {
            if (i > 0) {
                print_output_buffer[print_output_length++] = ' ';
            }
            memcpy(print_output_buffer + print_output_length, temp, temp_len);
            print_output_length += temp_len;
            print_output_buffer[print_output_length] = '\0';
        }
    }

    // Append this line to the accumulated output buffer
    if (all_output_length + print_output_length + 2 < sizeof(all_print_output)) {
        if (all_output_length > 0) {
            all_print_output[all_output_length++] = '\n';
        }
        memcpy(all_print_output + all_output_length, print_output_buffer, print_output_length);
        all_output_length += print_output_length;
        all_print_output[all_output_length] = '\0';
    }

    // Also print to console for debugging
    printf("%s\n", print_output_buffer);

    // Call the output callback if one is registered (for real-time GUI updates)
    if (output_callback) {
        output_callback(print_output_buffer);
    }

    // Decrement pending native call counter and check if we're done
    int remaining = __sync_sub_and_fetch(&pending_native_calls, 1);

    // If this was the last pending native call, VM has finished, and we have a completion callback
    if (remaining == 0 && vm_has_finished && completion_callback) {
        completion_callback(vm_exit_code);
    }

    return value_make_nil();
}

void wrapper_set_output_callback(OutputCallback callback) {
    output_callback = callback;
}

void wrapper_set_completion_callback(CompletionCallback callback) {
    completion_callback = callback;
}

const char* wrapper_get_last_print_output(void) {
    return print_output_length > 0 ? print_output_buffer : NULL;
}

const char* wrapper_get_all_print_output(void) {
    return all_output_length > 0 ? all_print_output : NULL;
}

void wrapper_clear_print_output(void) {
    print_output_length = 0;
    print_output_buffer[0] = '\0';
    all_output_length = 0;
    all_print_output[0] = '\0';
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  LED Matrix Hardware Simulation - C API for Rust integration               */
/* ──────────────────────────────────────────────────────────────────────────── */

// Forward declarations for Rust functions (implemented in Rust, called from C)
extern int rust_led_matrix_create(uint8_t matrix_id, uint16_t rows, uint16_t cols);
extern int rust_led_matrix_set_current(uint8_t matrix_id);
extern int rust_led_matrix_set_led(uint16_t row, uint16_t col, uint8_t r, uint8_t g, uint8_t b);
extern int rust_led_matrix_clear_all(void);
extern int rust_led_matrix_set_row_pattern(uint16_t row, uint32_t pattern, uint8_t r, uint8_t g, uint8_t b);
extern int rust_led_matrix_set_col_pattern(uint16_t col, uint32_t pattern, uint8_t r, uint8_t g, uint8_t b);
extern int rust_led_matrix_get_led_state(uint16_t row, uint16_t col);
extern int rust_led_matrix_delay_ms(uint32_t ms);

// C API wrapper functions (called from Rust)
int led_matrix_create(uint8_t matrix_id, uint16_t rows, uint16_t cols) {
    return rust_led_matrix_create(matrix_id, rows, cols);
}

int led_matrix_set_current(uint8_t matrix_id) {
    return rust_led_matrix_set_current(matrix_id);
}

int led_matrix_set_led(uint16_t row, uint16_t col, uint8_t r, uint8_t g, uint8_t b) {
    return rust_led_matrix_set_led(row, col, r, g, b);
}

int led_matrix_clear_all(void) {
    return rust_led_matrix_clear_all();
}

int led_matrix_set_row_pattern(uint16_t row, uint32_t pattern, uint8_t r, uint8_t g, uint8_t b) {
    return rust_led_matrix_set_row_pattern(row, pattern, r, g, b);
}

int led_matrix_set_col_pattern(uint16_t col, uint32_t pattern, uint8_t r, uint8_t g, uint8_t b) {
    return rust_led_matrix_set_col_pattern(col, pattern, r, g, b);
}

int led_matrix_get_led_state(uint16_t row, uint16_t col) {
    return rust_led_matrix_get_led_state(row, col);
}

int led_matrix_delay_ms(uint32_t ms) {
    return rust_led_matrix_delay_ms(ms);
}

/* ──────────────────────────────────────────────────────────────────────────── */
/*  LED Matrix Native Functions for Smollu VM                                  */
/* ──────────────────────────────────────────────────────────────────────────── */

Value emulator_native_led_matrix_init(Value* args, uint8_t argc) {
    if (argc != 2) {
        printf("led_matrix_init: Invalid argument count (expected 2, got %d)\n", argc);
        return value_make_nil();
    }

    if (args[0].type != VAL_INT || args[1].type != VAL_INT) {
        printf("led_matrix_init: Arguments must be integers (rows, cols)\n");
        return value_make_nil();
    }

    int32_t rows = args[0].as.i;
    int32_t cols = args[1].as.i;

    if (rows < 1 || rows > 64 || cols < 1 || cols > 64) {
        printf("led_matrix_init: Matrix dimensions out of range (1-64)\n");
        return value_make_nil();
    }

    // Create matrix with ID 0 (default matrix)
    int result = led_matrix_create(0, (uint16_t)rows, (uint16_t)cols);
    if (result != 0) {
        printf("led_matrix_init: Failed to create matrix\n");
    } else {
        printf("LED Matrix initialized: %dx%d\n", rows, cols);
    }

    return value_make_nil();
}

Value emulator_native_led_set(Value* args, uint8_t argc) {
    if (argc != 3) {
        printf("led_set: Invalid argument count (expected 3, got %d)\n", argc);
        return value_make_nil();
    }

    if (args[0].type != VAL_INT || args[1].type != VAL_INT || args[2].type != VAL_INT) {
        printf("led_set: Arguments must be integers (row, col, state)\n");
        return value_make_nil();
    }

    int32_t row = args[0].as.i;
    int32_t col = args[1].as.i;
    int32_t state = args[2].as.i;

    if (row < 0 || col < 0) {
        printf("led_set: Invalid coordinates (%d, %d)\n", row, col);
        return value_make_nil();
    }

    // Set LED: state 0 = off (black), state != 0 = on (white)
    uint8_t r = (state != 0) ? 255 : 0;
    uint8_t g = (state != 0) ? 255 : 0;
    uint8_t b = (state != 0) ? 255 : 0;

    int result = led_matrix_set_led((uint16_t)row, (uint16_t)col, r, g, b);
    if (result != 0) {
        printf("led_set: Failed to set LED at (%d, %d)\n", row, col);
    }

    return value_make_nil();
}

Value emulator_native_led_set_color(Value* args, uint8_t argc) {
    if (argc != 5) {
        printf("led_set_color: Invalid argument count (expected 5, got %d)\n", argc);
        return value_make_nil();
    }

    for (int i = 0; i < 5; i++) {
        if (args[i].type != VAL_INT) {
            printf("led_set_color: All arguments must be integers (row, col, r, g, b)\n");
            return value_make_nil();
        }
    }

    int32_t row = args[0].as.i;
    int32_t col = args[1].as.i;
    int32_t r = args[2].as.i;
    int32_t g = args[3].as.i;
    int32_t b = args[4].as.i;

    if (row < 0 || col < 0) {
        printf("led_set_color: Invalid coordinates (%d, %d)\n", row, col);
        return value_make_nil();
    }

    // Clamp color values to 0-255
    r = r < 0 ? 0 : (r > 255 ? 255 : r);
    g = g < 0 ? 0 : (g > 255 ? 255 : g);
    b = b < 0 ? 0 : (b > 255 ? 255 : b);

    int result = led_matrix_set_led((uint16_t)row, (uint16_t)col, (uint8_t)r, (uint8_t)g, (uint8_t)b);
    if (result != 0) {
        printf("led_set_color: Failed to set LED at (%d, %d)\n", row, col);
    }

    return value_make_nil();
}

Value emulator_native_led_clear(Value* args, uint8_t argc) {
    (void)args; // Unused parameter

    if (argc != 0) {
        printf("led_clear: Invalid argument count (expected 0, got %d)\n", argc);
        return value_make_nil();
    }

    int result = led_matrix_clear_all();
    if (result != 0) {
        printf("led_clear: Failed to clear matrix\n");
    } else {
        printf("LED Matrix cleared\n");
    }

    return value_make_nil();
}

Value emulator_native_led_set_row(Value* args, uint8_t argc) {
    if (argc != 2) {
        printf("led_set_row: Invalid argument count (expected 2, got %d)\n", argc);
        return value_make_nil();
    }

    if (args[0].type != VAL_INT || args[1].type != VAL_INT) {
        printf("led_set_row: Arguments must be integers (row, pattern)\n");
        return value_make_nil();
    }

    int32_t row = args[0].as.i;
    int32_t pattern = args[1].as.i;

    if (row < 0) {
        printf("led_set_row: Invalid row (%d)\n", row);
        return value_make_nil();
    }

    // Use white color for set bits
    int result = led_matrix_set_row_pattern((uint16_t)row, (uint32_t)pattern, 255, 255, 255);
    if (result != 0) {
        printf("led_set_row: Failed to set row pattern\n");
    }

    return value_make_nil();
}

Value emulator_native_led_set_col(Value* args, uint8_t argc) {
    if (argc != 2) {
        printf("led_set_col: Invalid argument count (expected 2, got %d)\n", argc);
        return value_make_nil();
    }

    if (args[0].type != VAL_INT || args[1].type != VAL_INT) {
        printf("led_set_col: Arguments must be integers (col, pattern)\n");
        return value_make_nil();
    }

    int32_t col = args[0].as.i;
    int32_t pattern = args[1].as.i;

    if (col < 0) {
        printf("led_set_col: Invalid column (%d)\n", col);
        return value_make_nil();
    }

    // Use white color for set bits
    int result = led_matrix_set_col_pattern((uint16_t)col, (uint32_t)pattern, 255, 255, 255);
    if (result != 0) {
        printf("led_set_col: Failed to set column pattern\n");
    }

    return value_make_nil();
}

Value emulator_native_led_get(Value* args, uint8_t argc) {
    if (argc != 2) {
        printf("led_get: Invalid argument count (expected 2, got %d)\n", argc);
        return value_from_int(0);
    }

    if (args[0].type != VAL_INT || args[1].type != VAL_INT) {
        printf("led_get: Arguments must be integers (row, col)\n");
        return value_from_int(0);
    }

    int32_t row = args[0].as.i;
    int32_t col = args[1].as.i;

    if (row < 0 || col < 0) {
        printf("led_get: Invalid coordinates (%d, %d)\n", row, col);
        return value_from_int(0);
    }

    int state = led_matrix_get_led_state((uint16_t)row, (uint16_t)col);
    return value_from_int(state);
}

Value emulator_native_delay_ms(Value* args, uint8_t argc) {
    if (argc != 1) {
        printf("delay_ms: Invalid argument count (expected 1, got %d)\n", argc);
        return value_make_nil();
    }

    if (args[0].type != VAL_INT) {
        printf("delay_ms: Argument must be an integer (ms)\n");
        return value_make_nil();
    }

    int32_t ms = args[0].as.i;

    if (ms < 0) {
        printf("delay_ms: Invalid delay time (%d)\n", ms);
        return value_make_nil();
    }

    int result = led_matrix_delay_ms((uint32_t)ms);
    if (result != 0) {
        printf("delay_ms: Failed to delay\n");
    }

    return value_make_nil();
}