//! Hardware simulation module
//!
//! This module provides hardware simulation capabilities for the Smollu emulator,
//! including LED matrices, GPIO pins, sensors, and other embedded system components.

pub mod led_matrix;

pub use led_matrix::LedMatrix;
