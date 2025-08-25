//! LED Matrix Hardware Simulation
//!
//! This module provides a configurable 2D LED matrix simulation that can be controlled
//! through native functions in Smollu programs and visualized in the GUI.

use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

/// RGB color representation for LEDs
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LedColor {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl LedColor {
    pub const OFF: LedColor = LedColor { r: 0, g: 0, b: 0 };
    pub const RED: LedColor = LedColor { r: 255, g: 0, b: 0 };
    pub const GREEN: LedColor = LedColor { r: 0, g: 255, b: 0 };
    pub const BLUE: LedColor = LedColor { r: 0, g: 0, b: 255 };
    pub const WHITE: LedColor = LedColor {
        r: 255,
        g: 255,
        b: 255,
    };
    pub const YELLOW: LedColor = LedColor {
        r: 255,
        g: 255,
        b: 0,
    };
    pub const CYAN: LedColor = LedColor {
        r: 0,
        g: 255,
        b: 255,
    };
    pub const MAGENTA: LedColor = LedColor {
        r: 255,
        g: 0,
        b: 255,
    };

    /// Create a new LED color from RGB values
    pub fn new(r: u8, g: u8, b: u8) -> Self {
        LedColor { r, g, b }
    }

    /// Check if the LED is on (any color component > 0)
    pub fn is_on(&self) -> bool {
        self.r > 0 || self.g > 0 || self.b > 0
    }

    /// Convert to egui Color32 for GUI rendering
    pub fn to_egui_color(&self) -> egui::Color32 {
        egui::Color32::from_rgb(self.r, self.g, self.b)
    }
}

impl Default for LedColor {
    fn default() -> Self {
        LedColor::OFF
    }
}

/// Individual LED state
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Led {
    pub color: LedColor,
    pub brightness: f32, // 0.0 to 1.0
}

impl Led {
    /// Create a new LED in the off state
    pub fn new() -> Self {
        Led {
            color: LedColor::OFF,
            brightness: 1.0,
        }
    }

    /// Turn the LED on with a specific color
    pub fn on(color: LedColor) -> Self {
        Led {
            color,
            brightness: 1.0,
        }
    }

    /// Turn the LED off
    pub fn off() -> Self {
        Led {
            color: LedColor::OFF,
            brightness: 0.0,
        }
    }

    /// Check if the LED is on
    pub fn is_on(&self) -> bool {
        self.color.is_on() && self.brightness > 0.0
    }

    /// Set the LED color and turn it on
    pub fn set_color(&mut self, color: LedColor) {
        self.color = color;
        if color.is_on() {
            self.brightness = 1.0;
        } else {
            self.brightness = 0.0;
        }
    }

    /// Set the LED brightness (0.0 to 1.0)
    pub fn set_brightness(&mut self, brightness: f32) {
        self.brightness = brightness.clamp(0.0, 1.0);
        if brightness <= 0.0 {
            self.color = LedColor::OFF;
        }
    }

    /// Get the final color taking brightness into account
    pub fn final_color(&self) -> LedColor {
        if self.brightness <= 0.0 {
            return LedColor::OFF;
        }

        LedColor::new(
            (self.color.r as f32 * self.brightness) as u8,
            (self.color.g as f32 * self.brightness) as u8,
            (self.color.b as f32 * self.brightness) as u8,
        )
    }
}

impl Default for Led {
    fn default() -> Self {
        Self::new()
    }
}

/// Delay state for hardware timing simulation
#[derive(Debug, Clone)]
struct DelayState {
    delay_until: Option<Instant>,
    delay_duration: Duration,
}

impl DelayState {
    fn new() -> Self {
        DelayState {
            delay_until: None,
            delay_duration: Duration::from_millis(0),
        }
    }

    fn start_delay(&mut self, ms: u32) {
        self.delay_duration = Duration::from_millis(ms as u64);
        self.delay_until = Some(Instant::now() + self.delay_duration);
    }

    fn is_delayed(&self) -> bool {
        if let Some(until) = self.delay_until {
            Instant::now() < until
        } else {
            false
        }
    }

    fn clear_delay(&mut self) {
        self.delay_until = None;
        self.delay_duration = Duration::from_millis(0);
    }

    fn remaining_delay(&self) -> Duration {
        if let Some(until) = self.delay_until {
            if Instant::now() < until {
                until.duration_since(Instant::now())
            } else {
                Duration::from_millis(0)
            }
        } else {
            Duration::from_millis(0)
        }
    }
}

/// 2D LED Matrix for hardware simulation
#[derive(Debug, Clone)]
pub struct LedMatrix {
    rows: usize,
    cols: usize,
    leds: Vec<Vec<Led>>,
    update_count: u64,       // For tracking changes in GUI
    delay_state: DelayState, // For non-blocking delay simulation
}

impl LedMatrix {
    /// Create a new LED matrix with specified dimensions
    pub fn new(rows: usize, cols: usize) -> Self {
        let leds = vec![vec![Led::new(); cols]; rows];
        LedMatrix {
            rows,
            cols,
            leds,
            update_count: 0,
            delay_state: DelayState::new(),
        }
    }

    /// Get matrix dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Check if coordinates are valid
    pub fn is_valid_coord(&self, row: usize, col: usize) -> bool {
        row < self.rows && col < self.cols
    }

    /// Get LED at specific coordinates
    pub fn get_led(&self, row: usize, col: usize) -> Option<&Led> {
        if self.is_valid_coord(row, col) {
            Some(&self.leds[row][col])
        } else {
            None
        }
    }

    /// Set LED at specific coordinates
    pub fn set_led(&mut self, row: usize, col: usize, led: Led) -> bool {
        if self.is_valid_coord(row, col) {
            self.leds[row][col] = led;
            self.update_count += 1;
            true
        } else {
            false
        }
    }

    /// Turn on LED with specific color
    pub fn turn_on(&mut self, row: usize, col: usize, color: LedColor) -> bool {
        self.set_led(row, col, Led::on(color))
    }

    /// Turn off LED
    pub fn turn_off(&mut self, row: usize, col: usize) -> bool {
        self.set_led(row, col, Led::off())
    }

    /// Clear all LEDs (turn them off)
    pub fn clear(&mut self) {
        for row in &mut self.leds {
            for led in row {
                *led = Led::off();
            }
        }
        self.update_count += 1;
    }

    /// Set an entire row of LEDs based on a bit pattern
    pub fn set_row(&mut self, row: usize, pattern: u32, color: LedColor) -> bool {
        if row >= self.rows {
            return false;
        }

        for col in 0..self.cols {
            let bit_set = (pattern & (1 << col)) != 0;
            if bit_set {
                self.leds[row][col] = Led::on(color);
            } else {
                self.leds[row][col] = Led::off();
            }
        }
        self.update_count += 1;
        true
    }

    /// Set an entire column of LEDs based on a bit pattern
    pub fn set_col(&mut self, col: usize, pattern: u32, color: LedColor) -> bool {
        if col >= self.cols {
            return false;
        }

        for row in 0..self.rows {
            let bit_set = (pattern & (1 << row)) != 0;
            if bit_set {
                self.leds[row][col] = Led::on(color);
            } else {
                self.leds[row][col] = Led::off();
            }
        }
        self.update_count += 1;
        true
    }

    /// Get all LEDs as a flat iterator (row-major order)
    pub fn iter_leds(&self) -> impl Iterator<Item = &Led> {
        self.leds.iter().flat_map(|row| row.iter())
    }

    /// Get all LEDs with their coordinates
    pub fn iter_with_coords(&self) -> impl Iterator<Item = (usize, usize, &Led)> {
        self.leds.iter().enumerate().flat_map(|(row, led_row)| {
            led_row
                .iter()
                .enumerate()
                .map(move |(col, led)| (row, col, led))
        })
    }

    /// Get the update count (for GUI change detection)
    pub fn update_count(&self) -> u64 {
        self.update_count
    }

    /// Resize the matrix (preserving existing LED states where possible)
    pub fn resize(&mut self, new_rows: usize, new_cols: usize) {
        // Create new matrix
        let mut new_leds = vec![vec![Led::new(); new_cols]; new_rows];

        // Copy existing LEDs
        let copy_rows = self.rows.min(new_rows);
        let copy_cols = self.cols.min(new_cols);

        for (row, new_row) in new_leds.iter_mut().enumerate().take(copy_rows) {
            for (col, new_led) in new_row.iter_mut().enumerate().take(copy_cols) {
                *new_led = self.leds[row][col];
            }
        }

        self.rows = new_rows;
        self.cols = new_cols;
        self.leds = new_leds;
        self.update_count += 1;
        // Note: delay_state is preserved automatically
    }

    /// Start a non-blocking delay for a specified number of milliseconds
    pub fn delay_ms(&mut self, ms: u32) {
        self.delay_state.start_delay(ms);
    }

    /// Check if the matrix is currently in a delay state
    pub fn is_delayed(&self) -> bool {
        self.delay_state.is_delayed()
    }

    /// Get remaining delay time in milliseconds
    pub fn remaining_delay_ms(&self) -> u32 {
        self.delay_state.remaining_delay().as_millis() as u32
    }

    /// Clear any active delay
    pub fn clear_delay(&mut self) {
        self.delay_state.clear_delay();
    }
}

/// Delay coordination system for synchronizing VM and GUI threads
pub struct DelayCoordination {
    pub delay_request_sender: Option<mpsc::Sender<u32>>, // Send delay requests to GUI
    pub delay_completion_receiver: Option<mpsc::Receiver<()>>, // Receive completion signals from GUI
}

impl DelayCoordination {
    fn new() -> Self {
        DelayCoordination {
            delay_request_sender: None,
            delay_completion_receiver: None,
        }
    }

    /// Set the delay communication channels
    pub fn set_channels(
        &mut self,
        request_sender: mpsc::Sender<u32>,
        completion_receiver: mpsc::Receiver<()>,
    ) {
        self.delay_request_sender = Some(request_sender);
        self.delay_completion_receiver = Some(completion_receiver);
    }

    /// Request a delay and wait for completion (blocking for VM thread)
    pub fn request_delay_and_wait(&mut self, ms: u32) -> Result<(), String> {
        if let Some(ref sender) = self.delay_request_sender {
            // Send delay request to GUI
            sender
                .send(ms)
                .map_err(|_| "Failed to send delay request")?;

            // Wait for completion signal from GUI
            if let Some(ref receiver) = self.delay_completion_receiver {
                receiver
                    .recv()
                    .map_err(|_| "Failed to receive delay completion signal")?;
                Ok(())
            } else {
                Err("No completion receiver available".to_string())
            }
        } else {
            Err("No request sender available".to_string())
        }
    }
}

/// Thread-safe LED matrix manager for global access from C functions
pub struct LedMatrixManager {
    matrices: Arc<Mutex<HashMap<u8, LedMatrix>>>, // ID -> Matrix mapping
    current_matrix: Arc<Mutex<Option<u8>>>,       // Currently active matrix ID
    delay_coordination: Arc<Mutex<DelayCoordination>>, // Delay synchronization
}

impl LedMatrixManager {
    /// Create a new LED matrix manager
    pub fn new() -> Self {
        LedMatrixManager {
            matrices: Arc::new(Mutex::new(HashMap::new())),
            current_matrix: Arc::new(Mutex::new(None)),
            delay_coordination: Arc::new(Mutex::new(DelayCoordination::new())),
        }
    }

    /// Create a new LED matrix with given ID and dimensions
    pub fn create_matrix(&self, id: u8, rows: usize, cols: usize) -> bool {
        if let Ok(mut matrices) = self.matrices.lock() {
            matrices.insert(id, LedMatrix::new(rows, cols));

            // Set as current matrix if none is set
            if let Ok(mut current) = self.current_matrix.lock() {
                if current.is_none() {
                    *current = Some(id);
                }
            }
            true
        } else {
            false
        }
    }

    /// Set the current active matrix
    pub fn set_current_matrix(&self, id: u8) -> bool {
        if let Ok(matrices) = self.matrices.lock() {
            if matrices.contains_key(&id) {
                if let Ok(mut current) = self.current_matrix.lock() {
                    *current = Some(id);
                    return true;
                }
            }
        }
        false
    }

    /// Get current matrix ID
    pub fn get_current_matrix_id(&self) -> Option<u8> {
        *self.current_matrix.lock().ok()?
    }

    /// Execute an operation on the current matrix
    pub fn with_current_matrix<F, R>(&self, f: F) -> Option<R>
    where
        F: FnOnce(&mut LedMatrix) -> R,
    {
        let current_id = *self.current_matrix.lock().ok()?;
        let mut matrices = self.matrices.lock().ok()?;
        let matrix = matrices.get_mut(&current_id.unwrap_or(0))?;
        Some(f(matrix))
    }

    /// Get a clone of the current matrix for GUI display
    pub fn get_current_matrix_clone(&self) -> Option<LedMatrix> {
        let current_id = *self.current_matrix.lock().ok()?;
        let matrices = self.matrices.lock().ok()?;
        matrices.get(&current_id.unwrap_or(0)).cloned()
    }

    /// Get all matrix IDs
    pub fn get_matrix_ids(&self) -> Vec<u8> {
        if let Ok(matrices) = self.matrices.lock() {
            matrices.keys().cloned().collect()
        } else {
            Vec::new()
        }
    }

    /// Set delay coordination channels
    pub fn set_delay_channels(
        &self,
        request_sender: mpsc::Sender<u32>,
        completion_receiver: mpsc::Receiver<()>,
    ) -> Result<(), String> {
        if let Ok(mut coordination) = self.delay_coordination.lock() {
            coordination.set_channels(request_sender, completion_receiver);
            Ok(())
        } else {
            Err("Failed to lock delay coordination".to_string())
        }
    }

    /// Request a synchronized delay (blocks VM thread until GUI signals completion)
    pub fn request_synchronized_delay(&self, ms: u32) -> Result<(), String> {
        if let Ok(mut coordination) = self.delay_coordination.lock() {
            coordination.request_delay_and_wait(ms)
        } else {
            Err("Failed to lock delay coordination".to_string())
        }
    }
}

impl Default for LedMatrixManager {
    fn default() -> Self {
        Self::new()
    }
}

// Global LED matrix manager instance
use std::sync::OnceLock;
static GLOBAL_LED_MANAGER: OnceLock<LedMatrixManager> = OnceLock::new();

/// Get the global LED matrix manager
pub fn get_led_matrix_manager() -> &'static LedMatrixManager {
    GLOBAL_LED_MANAGER.get_or_init(LedMatrixManager::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_led_color_creation() {
        let red = LedColor::RED;
        assert_eq!(red.r, 255);
        assert_eq!(red.g, 0);
        assert_eq!(red.b, 0);
        assert!(red.is_on());

        let off = LedColor::OFF;
        assert!(!off.is_on());
    }

    #[test]
    fn test_led_operations() {
        let mut led = Led::new();
        assert!(!led.is_on());

        led.set_color(LedColor::BLUE);
        assert!(led.is_on());
        assert_eq!(led.color, LedColor::BLUE);

        led.set_brightness(0.5);
        assert_eq!(led.brightness, 0.5);

        let final_color = led.final_color();
        assert_eq!(final_color.b, 127); // 255 * 0.5 = 127.5 -> 127
    }

    #[test]
    fn test_led_matrix_basic_operations() {
        let mut matrix = LedMatrix::new(8, 8);
        assert_eq!(matrix.dimensions(), (8, 8));

        // Test setting and getting LEDs
        assert!(matrix.turn_on(0, 0, LedColor::RED));
        let led = matrix.get_led(0, 0).unwrap();
        assert!(led.is_on());
        assert_eq!(led.color, LedColor::RED);

        // Test invalid coordinates
        assert!(!matrix.turn_on(10, 10, LedColor::GREEN));
        assert!(matrix.get_led(10, 10).is_none());

        // Test clear
        matrix.clear();
        let led = matrix.get_led(0, 0).unwrap();
        assert!(!led.is_on());
    }

    #[test]
    fn test_led_matrix_pattern_operations() {
        let mut matrix = LedMatrix::new(4, 4);

        // Test row pattern (binary 1010 = LEDs at positions 1 and 3)
        assert!(matrix.set_row(0, 0b1010, LedColor::GREEN));

        assert!(!matrix.get_led(0, 0).unwrap().is_on());
        assert!(matrix.get_led(0, 1).unwrap().is_on());
        assert!(!matrix.get_led(0, 2).unwrap().is_on());
        assert!(matrix.get_led(0, 3).unwrap().is_on());

        // Test column pattern
        assert!(matrix.set_col(0, 0b1100, LedColor::BLUE));

        assert!(!matrix.get_led(0, 0).unwrap().is_on());
        assert!(!matrix.get_led(1, 0).unwrap().is_on());
        assert!(matrix.get_led(2, 0).unwrap().is_on());
        assert!(matrix.get_led(3, 0).unwrap().is_on());
    }

    #[test]
    fn test_led_matrix_manager() {
        let manager = LedMatrixManager::new();

        // Create matrix
        assert!(manager.create_matrix(1, 8, 8));
        assert_eq!(manager.get_current_matrix_id(), Some(1));

        // Test operations on current matrix
        let result = manager.with_current_matrix(|matrix| {
            matrix.turn_on(0, 0, LedColor::RED);
            matrix.get_led(0, 0).unwrap().is_on()
        });
        assert_eq!(result, Some(true));

        // Test getting matrix clone
        let matrix_clone = manager.get_current_matrix_clone();
        assert!(matrix_clone.is_some());
        let matrix = matrix_clone.unwrap();
        assert!(matrix.get_led(0, 0).unwrap().is_on());
    }
}
