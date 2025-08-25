//! Smollu Emulator Library
//!
//! A Rust-based emulator for the Smollu VM with GUI capabilities for hardware simulation.
//! This library provides safe Rust bindings over the C VM implementation and includes
//! emulator-specific features like hardware simulation and debugging tools.

pub mod gui;
pub mod hardware;
pub mod vm;

pub use gui::SmolluEmulatorApp;
pub use hardware::LedMatrix;
pub use vm::{SmolluVM, Value, ValueType, VmError};

use anyhow::Result;
use hardware::led_matrix::{get_led_matrix_manager, LedMatrix as HardwareLedMatrix};
use std::fs;
use std::path::Path;

/// Emulator for the Smollu VM with additional debugging and simulation capabilities
pub struct SmolluEmulator {
    vm: SmolluVM,
    output_history: Vec<String>,
    bytecode_data: Option<Vec<u8>>, // Keep bytecode alive for VM
}

impl SmolluEmulator {
    /// Create a new emulator instance
    pub fn new() -> Result<Self, VmError> {
        let vm = SmolluVM::new()?;
        Ok(SmolluEmulator {
            vm,
            output_history: Vec::new(),
            bytecode_data: None,
        })
    }

    /// Load bytecode from a file
    pub fn load_bytecode_file<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let bytecode = fs::read(path)?;
        self.vm.load_bytecode(&bytecode)?;
        self.bytecode_data = Some(bytecode); // Keep bytecode alive
        Ok(())
    }

    /// Load bytecode from a byte slice
    pub fn load_bytecode(&mut self, bytecode: &[u8]) -> Result<(), VmError> {
        self.vm.load_bytecode(bytecode)?;
        self.bytecode_data = None; // External bytecode, don't store a copy
        Ok(())
    }

    /// Run the VM and capture output
    pub fn run(&mut self) -> Result<i32, VmError> {
        self.vm.clear_print_output();
        let result = self.vm.run()?;

        // Capture any output from the run
        if let Some(output) = self.vm.get_last_print_output() {
            self.output_history.push(output);
        }

        Ok(result)
    }

    /// Reset the VM state
    pub fn reset(&mut self) {
        self.vm.reset();
        self.output_history.clear();
        // Note: We keep bytecode_data as it may be needed for the VM pointer
    }

    /// Get all captured output from print statements
    pub fn get_output_history(&self) -> &[String] {
        &self.output_history
    }

    /// Get the current VM state for debugging
    pub fn get_vm_state(&self) -> VmState {
        VmState {
            pc: self.vm.get_pc(),
            sp: self.vm.get_sp(),
            stack_top: if self.vm.get_sp() > 0 {
                Some(self.vm.get_stack_value(0))
            } else {
                None
            },
        }
    }

    /// Get a global variable value
    pub fn get_global(&self, slot: u8) -> Value {
        self.vm.get_global(slot)
    }

    /// Set a global variable value
    pub fn set_global(&mut self, slot: u8, value: Value) {
        self.vm.set_global(slot, value);
    }

    /// Clear the output history
    pub fn clear_output_history(&mut self) {
        self.output_history.clear();
    }

    /// Set a callback for real-time output during VM execution
    pub fn set_output_callback(&self, sender: std::sync::mpsc::Sender<String>) {
        self.vm.set_output_callback(sender);
    }

    /// Set a callback for VM execution completion notifications
    pub fn set_completion_callback(&self, sender: std::sync::mpsc::Sender<i32>) {
        self.vm.set_completion_callback(sender);
    }

    /// Get all accumulated print output from the VM
    pub fn get_all_print_output(&self) -> Option<String> {
        self.vm.get_all_print_output()
    }

    /// Get the current LED matrix for GUI display
    pub fn get_led_matrix(&self) -> Option<HardwareLedMatrix> {
        let manager = get_led_matrix_manager();
        manager.get_current_matrix_clone()
    }

    /// Create a new LED matrix with given dimensions
    pub fn create_led_matrix(&self, rows: usize, cols: usize) -> bool {
        let manager = get_led_matrix_manager();
        manager.create_matrix(0, rows, cols)
    }

    /// Check if an LED matrix exists
    pub fn has_led_matrix(&self) -> bool {
        self.get_led_matrix().is_some()
    }
}

/// VM state information for debugging
#[derive(Debug, Clone)]
pub struct VmState {
    pub pc: usize,
    pub sp: u8,
    pub stack_top: Option<Value>,
}

impl Default for SmolluEmulator {
    fn default() -> Self {
        Self::new().expect("Failed to create emulator")
    }
}
