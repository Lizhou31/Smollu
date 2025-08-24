use egui::{Color32, RichText, ScrollArea, TextEdit, Ui};

pub struct ConsoleWidget {
    output_buffer: String,
    auto_scroll: bool,
}

impl Default for ConsoleWidget {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsoleWidget {
    pub fn new() -> Self {
        Self {
            output_buffer: String::new(),
            auto_scroll: true,
        }
    }

    pub fn add_output(&mut self, output: &str) {
        if !self.output_buffer.is_empty() {
            self.output_buffer.push('\n');
        }
        self.output_buffer.push_str(output);
    }

    pub fn clear(&mut self) {
        self.output_buffer.clear();
    }

    pub fn show(&mut self, ui: &mut Ui) {
        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Console Output").strong());
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("Clear").clicked() {
                            self.clear();
                        }
                        ui.checkbox(&mut self.auto_scroll, "Auto-scroll");
                    });
                });

                ui.separator();

                let available_height = ui.available_height() - 20.0;
                
                ScrollArea::vertical()
                    .max_height(available_height)
                    .auto_shrink([false, false])
                    .stick_to_bottom(self.auto_scroll)
                    .show(ui, |ui| {
                        if self.output_buffer.is_empty() {
                            ui.label(
                                RichText::new("No output yet...")
                                    .color(Color32::GRAY)
                                    .italics(),
                            );
                        } else {
                            ui.add(
                                TextEdit::multiline(&mut self.output_buffer.as_str())
                                    .desired_width(f32::INFINITY)
                                    .font(egui::TextStyle::Monospace)
                                    .interactive(false),
                            );
                        }
                    });
            });
        });
    }
}