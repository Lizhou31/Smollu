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
 * Get the last printed message (for GUI display)
 * @return Pointer to null-terminated string, or NULL if no output
 */
const char* wrapper_get_last_print_output(void);

/**
 * Clear the print output buffer
 */
void wrapper_clear_print_output(void);

#ifdef __cplusplus
}
#endif

#endif /* SMOLLU_WRAPPER_H */