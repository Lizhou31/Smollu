use egui::{Button, Color32, RichText, Ui};
use std::path::PathBuf;

#[derive(Debug, Clone)]
pub enum ControlAction {
    LoadFile,
    Run,
    Reset,
}

pub struct ControlsWidget {
    current_file: Option<PathBuf>,
    vm_running: bool,
    vm_loaded: bool,
}

impl Default for ControlsWidget {
    fn default() -> Self {
        Self::new()
    }
}

impl ControlsWidget {
    pub fn new() -> Self {
        Self {
            current_file: None,
            vm_running: false,
            vm_loaded: false,
        }
    }

    pub fn set_current_file(&mut self, file: Option<PathBuf>) {
        self.current_file = file;
        self.vm_loaded = self.current_file.is_some();
    }

    pub fn set_vm_running(&mut self, running: bool) {
        self.vm_running = running;
    }

    pub fn set_vm_loaded(&mut self, loaded: bool) {
        self.vm_loaded = loaded;
    }

    pub fn current_file(&self) -> &Option<PathBuf> {
        &self.current_file
    }

    pub fn show(&mut self, ui: &mut Ui) -> Option<ControlAction> {
        let mut action = None;

        ui.group(|ui| {
            ui.vertical(|ui| {
                ui.label(RichText::new("VM Controls").strong());
                ui.separator();

                // File info section
                ui.horizontal(|ui| {
                    ui.label("File:");
                    if let Some(ref file) = self.current_file {
                        ui.label(
                            RichText::new(
                                file.file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .as_ref(),
                            )
                            .color(Color32::LIGHT_BLUE),
                        );
                    } else {
                        ui.label(RichText::new("None").color(Color32::GRAY));
                    }
                });

                ui.add_space(10.0);

                // Control buttons
                ui.horizontal(|ui| {
                    if ui
                        .add(Button::new(RichText::new("📁 Load").size(14.0)))
                        .clicked()
                    {
                        action = Some(ControlAction::LoadFile);
                    }

                    let run_button = if self.vm_running {
                        Button::new(RichText::new("⏹ Stop").size(14.0).color(Color32::RED))
                    } else {
                        Button::new(RichText::new("▶ Run").size(14.0).color(Color32::GREEN))
                    };

                    if ui
                        .add_enabled(self.vm_loaded && !self.vm_running, run_button)
                        .clicked()
                    {
                        action = Some(ControlAction::Run);
                    }

                    if ui
                        .add_enabled(
                            self.vm_loaded,
                            Button::new(RichText::new("🔄 Reset").size(14.0)),
                        )
                        .clicked()
                    {
                        action = Some(ControlAction::Reset);
                    }
                });

                // Status section
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    ui.label("Status:");
                    let (status_text, status_color) = if self.vm_running {
                        ("Running", Color32::GREEN)
                    } else if self.vm_loaded {
                        ("Ready", Color32::LIGHT_BLUE)
                    } else {
                        ("No file loaded", Color32::GRAY)
                    };
                    ui.label(RichText::new(status_text).color(status_color));
                });
            });
        });

        action
    }
}
