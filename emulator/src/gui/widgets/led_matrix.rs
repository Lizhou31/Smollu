//! LED Matrix Widget for GUI visualization
//!
//! This widget provides a visual representation of the LED matrix hardware simulation,
//! displaying individual LEDs with their current colors and states.

use crate::hardware::led_matrix::LedMatrix;
use egui::{Color32, Rect, Response, Sense, Stroke, Ui, Vec2};

/// Configuration for LED matrix visualization
#[derive(Debug, Clone)]
pub struct LedMatrixConfig {
    /// Size of each LED in pixels
    pub led_size: f32,
    /// Spacing between LEDs in pixels
    pub led_spacing: f32,
    /// Color for LEDs that are off
    pub off_color: Color32,
    /// Border color for LED grid
    pub border_color: Color32,
    /// Whether to show grid lines
    pub show_grid: bool,
    /// Whether to show coordinates
    pub show_coordinates: bool,
}

impl Default for LedMatrixConfig {
    fn default() -> Self {
        LedMatrixConfig {
            led_size: 20.0,
            led_spacing: 2.0,
            off_color: Color32::DARK_GRAY,
            border_color: Color32::GRAY,
            show_grid: true,
            show_coordinates: false,
        }
    }
}

/// LED Matrix widget for displaying hardware simulation
pub struct LedMatrixWidget {
    config: LedMatrixConfig,
    matrix_cache: Option<LedMatrix>,
    last_update_count: u64,
    hover_pos: Option<(usize, usize)>,
}

impl LedMatrixWidget {
    /// Create a new LED matrix widget
    pub fn new() -> Self {
        LedMatrixWidget {
            config: LedMatrixConfig::default(),
            matrix_cache: None,
            last_update_count: 0,
            hover_pos: None,
        }
    }

    /// Create a new LED matrix widget with custom configuration
    pub fn with_config(config: LedMatrixConfig) -> Self {
        LedMatrixWidget {
            config,
            matrix_cache: None,
            last_update_count: 0,
            hover_pos: None,
        }
    }

    /// Update the configuration
    pub fn set_config(&mut self, config: LedMatrixConfig) {
        self.config = config;
    }

    /// Get the current configuration
    pub fn config(&self) -> &LedMatrixConfig {
        &self.config
    }

    /// Update the LED matrix data
    pub fn update_matrix(&mut self, matrix: Option<LedMatrix>) {
        if let Some(new_matrix) = matrix {
            // Only update if the matrix has actually changed
            if self.matrix_cache.is_none() || self.last_update_count != new_matrix.update_count() {
                self.last_update_count = new_matrix.update_count();
                self.matrix_cache = Some(new_matrix);
            }
        } else {
            self.matrix_cache = None;
            self.last_update_count = 0;
        }
    }

    /// Calculate the total size needed for the widget
    fn calculate_size(&self, rows: usize, cols: usize) -> Vec2 {
        let led_size = self.config.led_size;
        let spacing = self.config.led_spacing;

        let width = cols as f32 * (led_size + spacing) - spacing + 2.0 * spacing; // Add border
        let height = rows as f32 * (led_size + spacing) - spacing + 2.0 * spacing; // Add border

        Vec2::new(width, height)
    }

    /// Convert screen position to matrix coordinates
    fn screen_to_matrix_coords(
        &self,
        pos: egui::Pos2,
        widget_rect: Rect,
        rows: usize,
        cols: usize,
    ) -> Option<(usize, usize)> {
        let led_size = self.config.led_size;
        let spacing = self.config.led_spacing;

        // Calculate relative position within the widget
        let rel_pos = pos - widget_rect.min;

        // Adjust for border spacing
        let adj_pos = egui::Pos2::new(rel_pos.x - spacing, rel_pos.y - spacing);

        // Calculate grid position
        let grid_x = adj_pos.x / (led_size + spacing);
        let grid_y = adj_pos.y / (led_size + spacing);

        let col = grid_x.floor() as isize;
        let row = grid_y.floor() as isize;

        // Check bounds
        if row >= 0 && row < rows as isize && col >= 0 && col < cols as isize {
            Some((row as usize, col as usize))
        } else {
            None
        }
    }

    /// Show the LED matrix widget
    pub fn show(&mut self, ui: &mut Ui) -> Response {
        // Check if we have a matrix to display
        let matrix = match &self.matrix_cache {
            Some(matrix) => matrix,
            None => {
                // Show placeholder when no matrix is available
                return ui.label("No LED matrix initialized. Use 'led_matrix_init(rows, cols)' in your Smollu program.");
            }
        };

        let (rows, cols) = matrix.dimensions();
        let widget_size = self.calculate_size(rows, cols);

        // Reserve space for the widget
        let (response, painter) = ui.allocate_painter(widget_size, Sense::hover());
        let widget_rect = response.rect;

        // Update hover position
        if response.hovered() {
            if let Some(hover_pos) = response.hover_pos() {
                self.hover_pos = self.screen_to_matrix_coords(hover_pos, widget_rect, rows, cols);
            }
        } else {
            self.hover_pos = None;
        }

        // Draw background
        painter.rect_filled(widget_rect, 0.0, Color32::BLACK);

        // Draw LEDs
        let led_size = self.config.led_size;
        let spacing = self.config.led_spacing;
        let start_pos = widget_rect.min + Vec2::new(spacing, spacing);

        for (row, col, led) in matrix.iter_with_coords() {
            let led_pos = egui::Pos2::new(
                start_pos.x + col as f32 * (led_size + spacing),
                start_pos.y + row as f32 * (led_size + spacing),
            );

            let led_rect = Rect::from_min_size(led_pos, Vec2::splat(led_size));

            // Determine LED color
            let led_color = if led.is_on() {
                let final_color = led.final_color();
                Color32::from_rgb(final_color.r, final_color.g, final_color.b)
            } else {
                self.config.off_color
            };

            // Highlight hovered LED
            let is_hovered = self.hover_pos == Some((row, col));
            let border_color = if is_hovered {
                Color32::WHITE
            } else {
                self.config.border_color
            };

            // Draw LED with border
            painter.rect_filled(led_rect, 2.0, led_color);
            if self.config.show_grid || is_hovered {
                painter.rect_stroke(led_rect, 2.0, Stroke::new(1.0, border_color));
            }
        }

        // Draw grid lines if enabled
        if self.config.show_grid {
            // Vertical lines
            for col in 0..=cols {
                let x = start_pos.x + col as f32 * (led_size + spacing) - spacing / 2.0;
                let start = egui::Pos2::new(x, widget_rect.min.y);
                let end = egui::Pos2::new(x, widget_rect.max.y);
                painter.line_segment([start, end], Stroke::new(0.5, self.config.border_color));
            }

            // Horizontal lines
            for row in 0..=rows {
                let y = start_pos.y + row as f32 * (led_size + spacing) - spacing / 2.0;
                let start = egui::Pos2::new(widget_rect.min.x, y);
                let end = egui::Pos2::new(widget_rect.max.x, y);
                painter.line_segment([start, end], Stroke::new(0.5, self.config.border_color));
            }
        }

        // Show coordinates if enabled and there's a hovered LED
        if self.config.show_coordinates {
            if let Some((row, col)) = self.hover_pos {
                let tooltip_text = format!("LED ({}, {})", row, col);
                return response.on_hover_text(tooltip_text);
            }
        }

        response
    }

    /// Show the LED matrix with configuration controls
    pub fn show_with_controls(&mut self, ui: &mut Ui) -> Response {
        ui.vertical(|ui| {
            // Configuration controls
            ui.collapsing("LED Matrix Settings", |ui| {
                ui.horizontal(|ui| {
                    ui.label("LED Size:");
                    ui.add(egui::Slider::new(&mut self.config.led_size, 5.0..=50.0).suffix(" px"));
                });

                ui.horizontal(|ui| {
                    ui.label("Spacing:");
                    ui.add(
                        egui::Slider::new(&mut self.config.led_spacing, 0.0..=10.0).suffix(" px"),
                    );
                });

                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.config.show_grid, "Show grid");
                    ui.checkbox(
                        &mut self.config.show_coordinates,
                        "Show coordinates on hover",
                    );
                });

                ui.horizontal(|ui| {
                    ui.label("Off LED color:");
                    ui.color_edit_button_srgba(&mut self.config.off_color);

                    ui.label("Grid color:");
                    ui.color_edit_button_srgba(&mut self.config.border_color);
                });
            });

            ui.separator();

            // Matrix display
            let response = self.show(ui);

            // Status information
            if let Some(matrix) = &self.matrix_cache {
                let (rows, cols) = matrix.dimensions();
                let led_count = rows * cols;
                let on_count = matrix.iter_leds().filter(|led| led.is_on()).count();

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(format!("Matrix: {}×{} ({} LEDs)", rows, cols, led_count));
                    ui.separator();
                    ui.label(format!("On: {}/{}", on_count, led_count));

                    // Show delay status
                    if matrix.is_delayed() {
                        ui.separator();
                        let remaining_ms = matrix.remaining_delay_ms();
                        ui.colored_label(
                            egui::Color32::YELLOW,
                            format!("Delay: {}ms", remaining_ms),
                        );
                    }

                    if let Some((row, col)) = self.hover_pos {
                        ui.separator();
                        ui.label(format!("Hover: ({}, {})", row, col));
                    }
                });
            }

            response
        })
        .inner
    }

    /// Get the current hover position
    pub fn hover_position(&self) -> Option<(usize, usize)> {
        self.hover_pos
    }

    /// Check if a specific LED is being hovered
    pub fn is_led_hovered(&self, row: usize, col: usize) -> bool {
        self.hover_pos == Some((row, col))
    }
}

impl Default for LedMatrixWidget {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::led_matrix::{LedColor, LedMatrix};

    #[test]
    fn test_widget_creation() {
        let widget = LedMatrixWidget::new();
        assert!(widget.matrix_cache.is_none());
        assert_eq!(widget.last_update_count, 0);
        assert!(widget.hover_pos.is_none());
    }

    #[test]
    fn test_size_calculation() {
        let widget = LedMatrixWidget::new();
        let size = widget.calculate_size(8, 8);

        // Expected: 8 * (20 + 2) - 2 + 2*2 = 8 * 22 - 2 + 4 = 176 - 2 + 4 = 178
        assert_eq!(size.x, 178.0);
        assert_eq!(size.y, 178.0);
    }

    #[test]
    fn test_matrix_update() {
        let mut widget = LedMatrixWidget::new();
        let mut matrix = LedMatrix::new(4, 4);

        // Initial update
        widget.update_matrix(Some(matrix.clone()));
        assert!(widget.matrix_cache.is_some());
        let initial_count = widget.last_update_count;

        // Update without changes should not trigger cache update
        widget.update_matrix(Some(matrix.clone()));
        assert_eq!(widget.last_update_count, initial_count);

        // Update with changes should trigger cache update
        matrix.turn_on(0, 0, LedColor::RED);
        widget.update_matrix(Some(matrix));
        assert!(widget.last_update_count > initial_count);
    }

    #[test]
    fn test_config_modification() {
        let mut widget = LedMatrixWidget::new();
        let mut config = LedMatrixConfig::default();
        config.led_size = 25.0;
        config.show_grid = false;

        widget.set_config(config.clone());
        assert_eq!(widget.config().led_size, 25.0);
        assert!(!widget.config().show_grid);
    }
}
