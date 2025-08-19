//! Safe Rust wrapper around the Smollu VM C API
//! 
//! This module provides safe, idiomatic Rust interfaces over the raw C bindings.
//! It handles memory management, error checking, and type safety.

use crate::vm::bindings::*;
use std::ffi::CStr;
use std::ptr;
use anyhow::Result;

/// Error types for VM operations
#[derive(Debug, thiserror::Error)]
pub enum VmError {
    #[error("VM creation failed")]
    CreationFailed,
    #[error("Invalid bytecode: {0}")]
    InvalidBytecode(String),
    #[error("VM execution error: {0}")]
    ExecutionError(i32),
    #[error("Invalid value type conversion")]
    InvalidValueType,
    #[error("Null pointer error")]
    NullPointer,
}

/// Safe wrapper around the Smollu VM
pub struct SmolluVM {
    vm_ptr: *mut crate::vm::bindings::SmolluVM,
}

impl SmolluVM {
    /// Create a new VM instance
    pub fn new() -> Result<Self, VmError> {
        let vm_ptr = unsafe { wrapper_vm_create() };
        if vm_ptr.is_null() {
            return Err(VmError::CreationFailed);
        }
        
        Ok(SmolluVM { vm_ptr })
    }
    
    /// Load bytecode into the VM
    pub fn load_bytecode(&mut self, bytecode: &[u8]) -> Result<(), VmError> {
        let result = unsafe {
            wrapper_vm_load_bytecode(self.vm_ptr, bytecode.as_ptr(), bytecode.len())
        };
        
        match result {
            0 => Ok(()),
            -1 => Err(VmError::InvalidBytecode("Invalid parameters".to_string())),
            -2 => Err(VmError::InvalidBytecode("Bytecode too short or malformed".to_string())),
            _ => Err(VmError::InvalidBytecode(format!("Unknown error code: {}", result))),
        }
    }
    
    /// Run the VM until completion
    pub fn run(&mut self) -> Result<i32, VmError> {
        let result = unsafe { wrapper_vm_run(self.vm_ptr) };
        
        if result < 0 {
            Err(VmError::ExecutionError(result))
        } else {
            Ok(result)
        }
    }
    
    /// Reset the VM state (clear stack, reset PC, keep bytecode)
    pub fn reset(&mut self) {
        unsafe { wrapper_vm_reset(self.vm_ptr) };
    }
    
    /// Get current program counter
    pub fn get_pc(&self) -> usize {
        unsafe { wrapper_vm_get_pc(self.vm_ptr) }
    }
    
    /// Get current stack pointer (number of items on stack)
    pub fn get_sp(&self) -> u8 {
        unsafe { wrapper_vm_get_sp(self.vm_ptr) }
    }
    
    /// Get value from stack at given offset from top
    pub fn get_stack_value(&self, offset: u8) -> Value {
        let raw_value = unsafe { wrapper_vm_get_stack_value(self.vm_ptr, offset) };
        Value::from_raw(raw_value)
    }
    
    /// Get global variable value
    pub fn get_global(&self, slot: u8) -> Value {
        let raw_value = unsafe { wrapper_vm_get_global(self.vm_ptr, slot) };
        Value::from_raw(raw_value)
    }
    
    /// Set global variable value
    pub fn set_global(&mut self, slot: u8, value: Value) {
        let raw_value = value.to_raw();
        unsafe { wrapper_vm_set_global(self.vm_ptr, slot, raw_value) };
    }
    
    /// Get the last output from the print function
    pub fn get_last_print_output(&self) -> Option<String> {
        let output_ptr = unsafe { wrapper_get_last_print_output() };
        if output_ptr.is_null() {
            None
        } else {
            unsafe {
                let c_str = CStr::from_ptr(output_ptr);
                c_str.to_str().ok().map(|s| s.to_string())
            }
        }
    }
    
    /// Clear the print output buffer
    pub fn clear_print_output(&self) {
        unsafe { wrapper_clear_print_output() };
    }
}

impl Drop for SmolluVM {
    fn drop(&mut self) {
        if !self.vm_ptr.is_null() {
            unsafe { wrapper_vm_destroy(self.vm_ptr) };
            self.vm_ptr = ptr::null_mut();
        }
    }
}

// Make SmolluVM thread-safe by implementing Send and Sync
// SAFETY: The C VM is designed to be used from a single thread at a time,
// and our wrapper ensures exclusive access through &mut self for modifications
unsafe impl Send for SmolluVM {}
unsafe impl Sync for SmolluVM {}

/// Rust representation of Smollu values
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Nil,
    Bool(bool),
    Int(i32),
    Float(f32),
}

impl Value {
    /// Create Value from raw C Value struct
    fn from_raw(raw: crate::vm::bindings::Value) -> Self {
        match raw.type_ {
            crate::vm::bindings::ValueType_VAL_NIL => Value::Nil,
            crate::vm::bindings::ValueType_VAL_BOOL => {
                Value::Bool(unsafe { raw.as_.boolean })
            }
            crate::vm::bindings::ValueType_VAL_INT => {
                Value::Int(unsafe { raw.as_.i })
            }
            crate::vm::bindings::ValueType_VAL_FLOAT => {
                Value::Float(unsafe { raw.as_.f })
            }
            _ => Value::Nil, // Default to Nil for unknown types
        }
    }
    
    /// Convert Value to raw C Value struct
    fn to_raw(&self) -> crate::vm::bindings::Value {
        match self {
            Value::Nil => unsafe { wrapper_value_nil() },
            Value::Bool(b) => unsafe { wrapper_value_bool(*b) },
            Value::Int(i) => unsafe { wrapper_value_int(*i) },
            Value::Float(f) => unsafe { wrapper_value_float(*f) },
        }
    }
    
    /// Get the type of this value
    pub fn value_type(&self) -> ValueType {
        match self {
            Value::Nil => ValueType::Nil,
            Value::Bool(_) => ValueType::Bool,
            Value::Int(_) => ValueType::Int,
            Value::Float(_) => ValueType::Float,
        }
    }
    
    /// Check if value is truthy (used for conditionals)
    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Nil => false,
            Value::Bool(b) => *b,
            Value::Int(i) => *i != 0,
            Value::Float(f) => *f != 0.0,
        }
    }
    
    /// Convert value to display string
    pub fn to_string(&self) -> String {
        match self {
            Value::Nil => "nil".to_string(),
            Value::Bool(b) => if *b { "true".to_string() } else { "false".to_string() },
            Value::Int(i) => i.to_string(),
            Value::Float(f) => f.to_string(),
        }
    }
}

/// Enum representing the different value types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    Nil,
    Bool,
    Int,
    Float,
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}