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
    
    return 0;
}

int wrapper_vm_run(SmolluVM* vm) {
    if (!vm) {
        return -1;
    }
    
    return smollu_vm_run(vm);
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
    
    // Call the callback if one is registered (for real-time GUI updates)
    if (output_callback) {
        output_callback(print_output_buffer);
    }
    
    return value_make_nil();
}

void wrapper_set_output_callback(OutputCallback callback) {
    output_callback = callback;
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