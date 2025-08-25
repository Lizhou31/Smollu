/**
 * @file wrapper.h
 * @brief C wrapper functions for Smollu VM to provide safe FFI interface
 *
 * This header provides a simplified C interface over the Smollu VM that is
 * easier to bind to from Rust. It handles memory management and provides
 * safer function signatures.
 */

#ifndef SMOLLU_WRAPPER_H
#define SMOLLU_WRAPPER_H

#include "smollu_vm.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ──────────────────────────────────────────────────────────────────────────── */
/*  VM Lifecycle Management                                                     */
/* ──────────────────────────────────────────────────────────────────────────── */

/**
 * Create and initialize a new VM instance
 * @return Pointer to initialized VM, or NULL on failure
 */
SmolluVM* wrapper_vm_create(void);

/**
 * Destroy a VM instance and free its memory
 * @param vm VM instance to destroy
 */
void wrapper_vm_destroy(SmolluVM* vm);

/**
 * Load bytecode into the VM
 * @param vm VM instance
 * @param bytecode Bytecode buffer
 * @param length Length of bytecode buffer
 * @return 0 on success, negative on error
 */
int wrapper_vm_load_bytecode(SmolluVM* vm, const uint8_t* bytecode, size_t length);

/**
 * Run the VM until completion or error
 * @param vm VM instance
 * @return 0 on normal completion, positive for VM halt, negative on error
 */
int wrapper_vm_run(SmolluVM* vm);

/**
 * Reset VM state (clear stack, reset PC, keep bytecode loaded)
 * @param vm VM instance
 */
void wrapper_vm_reset(SmolluVM* vm);

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Native Function Management                                                  */
/* ──────────────────────────────────────────────────────────────────────────── */

/**
 * Register a native function with the VM
 * @param vm VM instance
 * @param function_id Native function ID (0-255)
 * @param function Function pointer
 * @return 0 on success, negative on error
 */
int wrapper_vm_register_native(SmolluVM* vm, uint8_t function_id, NativeFn function);

/* ──────────────────────────────────────────────────────────────────────────── */
/*  VM State Inspection                                                         */
/* ──────────────────────────────────────────────────────────────────────────── */

/**
 * Get current program counter
 * @param vm VM instance
 * @return Current PC value
 */
size_t wrapper_vm_get_pc(const SmolluVM* vm);

/**
 * Get current stack pointer
 * @param vm VM instance
 * @return Current stack pointer (number of items on stack)
 */
uint8_t wrapper_vm_get_sp(const SmolluVM* vm);

/**
 * Get value from stack at given offset from top
 * @param vm VM instance
 * @param offset Offset from stack top (0 = top)
 * @return Stack value, or NIL if offset is invalid
 */
Value wrapper_vm_get_stack_value(const SmolluVM* vm, uint8_t offset);

/**
 * Get global variable value
 * @param vm VM instance
 * @param slot Global slot number (0-255)
 * @return Global value
 */
Value wrapper_vm_get_global(const SmolluVM* vm, uint8_t slot);

/**
 * Set global variable value
 * @param vm VM instance
 * @param slot Global slot number (0-255)
 * @param value Value to set
 */
void wrapper_vm_set_global(SmolluVM* vm, uint8_t slot, Value value);

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Value Creation Helpers                                                      */
/* ──────────────────────────────────────────────────────────────────────────── */

/**
 * Create a NIL value
 * @return NIL value
 */
Value wrapper_value_nil(void);

/**
 * Create a boolean value
 * @param b Boolean value
 * @return Boolean Value
 */
Value wrapper_value_bool(bool b);

/**
 * Create an integer value
 * @param i Integer value
 * @return Integer Value
 */
Value wrapper_value_int(int32_t i);

/**
 * Create a float value
 * @param f Float value
 * @return Float Value
 */
Value wrapper_value_float(float f);

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Value Type Checking and Extraction                                          */
/* ──────────────────────────────────────────────────────────────────────────── */

/**
 * Get the type of a value
 * @param val Value to check
 * @return ValueType enum
 */
ValueType wrapper_value_type(Value val);

/**
 * Check if value is NIL
 * @param val Value to check
 * @return true if NIL, false otherwise
 */
bool wrapper_value_is_nil(Value val);

/**
 * Check if value is boolean
 * @param val Value to check
 * @return true if boolean, false otherwise
 */
bool wrapper_value_is_bool(Value val);

/**
 * Check if value is integer
 * @param val Value to check
 * @return true if integer, false otherwise
 */
bool wrapper_value_is_int(Value val);

/**
 * Check if value is float
 * @param val Value to check
 * @return true if float, false otherwise
 */
bool wrapper_value_is_float(Value val);

/**
 * Extract boolean value (only call if wrapper_value_is_bool returns true)
 * @param val Value to extract from
 * @return Boolean value
 */
bool wrapper_value_as_bool(Value val);

/**
 * Extract integer value (only call if wrapper_value_is_int returns true)
 * @param val Value to extract from
 * @return Integer value
 */
int32_t wrapper_value_as_int(Value val);

/**
 * Extract float value (only call if wrapper_value_is_float returns true)
 * @param val Value to extract from
 * @return Float value
 */
float wrapper_value_as_float(Value val);

/* ──────────────────────────────────────────────────────────────────────────── */
/*  Emulator-specific Native Functions                                          */
/* ──────────────────────────────────────────────────────────────────────────── */

/**
 * Native function: print (with emulator output capture)
 * This replaces the original print function to capture output for GUI display
 * @param args Function arguments
 * @param argc Number of arguments
 * @return NIL value
 */
Value emulator_native_print(Value* args, uint8_t argc);

/**
 * Callback function type for real-time output notifications
 * @param output The output string that was printed
 */
typedef void (*OutputCallback)(const char* output);

/**
 * Callback function type for VM execution completion notifications
 * Called when VM execution truly completes (including all native function calls)
 * @param exit_code The VM exit code
 */
typedef void (*CompletionCallback)(int exit_code);

/**
 * Set a callback function to be called whenever native_print is executed
 * @param callback Function to call with print output, or NULL to disable
 */
void wrapper_set_output_callback(OutputCallback callback);

/**
 * Set a callback function to be called when VM execution is truly complete
 * @param callback Function to call with completion status, or NULL to disable
 */
void wrapper_set_completion_callback(CompletionCallback callback);

/**
 * Get the last printed message (for GUI display)
 * @return Pointer to null-terminated string, or NULL if no output
 */
const char* wrapper_get_last_print_output(void);

/**
 * Get all accumulated print output (multiple calls)
 * @return Pointer to null-terminated string with all output, or NULL if no output
 */
const char* wrapper_get_all_print_output(void);

/**
 * Clear the print output buffer
 */
void wrapper_clear_print_output(void);

/* ──────────────────────────────────────────────────────────────────────────── */
/*  LED Matrix Hardware Simulation                                             */
/* ──────────────────────────────────────────────────────────────────────────── */

/**
 * Native function: led_matrix_init - Initialize LED matrix with given dimensions
 * @param args Function arguments [rows, cols]
 * @param argc Number of arguments (should be 2)
 * @return NIL value
 */
Value emulator_native_led_matrix_init(Value* args, uint8_t argc);

/**
 * Native function: led_set - Set individual LED state
 * @param args Function arguments [row, col, state] (state: 0=off, 1=on)
 * @param argc Number of arguments (should be 3)
 * @return NIL value
 */
Value emulator_native_led_set(Value* args, uint8_t argc);

/**
 * Native function: led_set_color - Set individual LED with color
 * @param args Function arguments [row, col, r, g, b] (r,g,b: 0-255)
 * @param argc Number of arguments (should be 5)
 * @return NIL value
 */
Value emulator_native_led_set_color(Value* args, uint8_t argc);

/**
 * Native function: led_clear - Clear all LEDs (turn them off)
 * @param args Function arguments (none)
 * @param argc Number of arguments (should be 0)
 * @return NIL value
 */
Value emulator_native_led_clear(Value* args, uint8_t argc);

/**
 * Native function: led_set_row - Set entire row based on bit pattern
 * @param args Function arguments [row, pattern] (pattern: bit mask)
 * @param argc Number of arguments (should be 2)
 * @return NIL value
 */
Value emulator_native_led_set_row(Value* args, uint8_t argc);

/**
 * Native function: led_set_col - Set entire column based on bit pattern
 * @param args Function arguments [col, pattern] (pattern: bit mask)
 * @param argc Number of arguments (should be 2)
 * @return NIL value
 */
Value emulator_native_led_set_col(Value* args, uint8_t argc);

/**
 * Native function: led_get - Get LED state
 * @param args Function arguments [row, col]
 * @param argc Number of arguments (should be 2)
 * @return Integer value (0=off, 1=on)
 */
Value emulator_native_led_get(Value* args, uint8_t argc);

/**
 * Native function: delay_ms - Delay for a specified number of milliseconds
 * @param args Function arguments [ms]
 * @param argc Number of arguments (should be 1)
 * @return NIL value
 */
Value emulator_native_delay_ms(Value* args, uint8_t argc);

/**
 * C API for LED matrix operations (called from Rust)
 */
int led_matrix_create(uint8_t matrix_id, uint16_t rows, uint16_t cols);
int led_matrix_set_current(uint8_t matrix_id);
int led_matrix_set_led(uint16_t row, uint16_t col, uint8_t r, uint8_t g, uint8_t b);
int led_matrix_clear_all(void);
int led_matrix_set_row_pattern(uint16_t row, uint32_t pattern, uint8_t r, uint8_t g, uint8_t b);
int led_matrix_set_col_pattern(uint16_t col, uint32_t pattern, uint8_t r, uint8_t g, uint8_t b);
int led_matrix_get_led_state(uint16_t row, uint16_t col);
int led_matrix_delay_ms(uint32_t ms);

#ifdef __cplusplus
}
#endif

#endif /* SMOLLU_WRAPPER_H */