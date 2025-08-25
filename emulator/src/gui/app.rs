use crate::gui::widgets::controls::ControlAction;
use crate::gui::widgets::{ConsoleWidget, ControlsWidget, LedMatrixWidget};
use crate::hardware::led_matrix::get_led_matrix_manager;
use crate::{SmolluEmulator, VmError};
use eframe::egui;
use egui::{Color32, RichText};
use rfd::FileDialog;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

pub struct SmolluEmulatorApp {
    emulator: SmolluEmulator,
    console: ConsoleWidget,
    controls: ControlsWidget,
    led_matrix: LedMatrixWidget,
    error_message: Option<String>,
    vm_execution_thread: Option<thread::JoinHandle<()>>,
    execution_receiver: Option<mpsc::Receiver<ExecutionResult>>,
    delay_completion_sender: Option<mpsc::Sender<()>>, // Signal delay completion to VM
    delay_start_time: Option<Instant>,                 // When current delay started
    delay_duration: Duration,                          // Total delay duration
}

#[derive(Debug)]
enum ExecutionResult {
    Output(String),
    Completed(i32),
    Error(String),
    DelayRequest(u32), // Delay request in milliseconds
}

impl Default for SmolluEmulatorApp {
    fn default() -> Self {
        Self::new()
    }
}

impl SmolluEmulatorApp {
    pub fn new() -> Self {
        let emulator = SmolluEmulator::new().expect("Failed to create emulator");

        Self {
            emulator,
            console: ConsoleWidget::new(),
            controls: ControlsWidget::new(),
            led_matrix: LedMatrixWidget::new(),
            error_message: None,
            vm_execution_thread: None,
            execution_receiver: None,
            delay_completion_sender: None,
            delay_start_time: None,
            delay_duration: Duration::from_millis(0),
        }
    }

    pub fn with_file(file_path: PathBuf) -> Self {
        let mut app = Self::new();
        app.load_file(file_path);
        app
    }

    fn load_file(&mut self, file_path: PathBuf) {
        match self.emulator.load_bytecode_file(&file_path) {
            Ok(()) => {
                self.controls.set_current_file(Some(file_path.clone()));
                self.console
                    .add_output(&format!("✅ Loaded bytecode from: {}", file_path.display()));
                self.error_message = None;
            }
            Err(e) => {
                self.error_message = Some(format!("Failed to load file: {}", e));
                self.controls.set_current_file(None);
            }
        }
    }

    fn run_vm(&mut self) {
        if self.vm_execution_thread.is_some() {
            return;
        }

        self.console.add_output("🚀 Starting VM execution...");

        let (tx, rx) = mpsc::channel();
        self.execution_receiver = Some(rx);

        let mut emulator_clone = SmolluEmulator::new().expect("Failed to create emulator clone");

        if let Some(current_file) = self.controls.current_file().clone() {
            if let Err(e) = emulator_clone.load_bytecode_file(&current_file) {
                self.error_message = Some(format!("Failed to reload bytecode: {}", e));
                self.controls.set_vm_running(false);
                return;
            }
        }

        // Set up real-time output callback
        let (output_tx, output_rx) = mpsc::channel();
        emulator_clone.set_output_callback(output_tx);

        // Set up completion callback
        let (completion_tx, completion_rx) = mpsc::channel();
        emulator_clone.set_completion_callback(completion_tx);

        // Set up delay coordination channels
        let (delay_completion_tx, delay_completion_rx) = mpsc::channel();
        self.delay_completion_sender = Some(delay_completion_tx);

        // Set up delay request channel (GUI receives delay requests from VM)
        let (delay_request_tx, delay_request_rx) = mpsc::channel();

        // Set delay channels on LED matrix manager
        let led_manager = get_led_matrix_manager();
        if let Err(e) = led_manager.set_delay_channels(delay_request_tx, delay_completion_rx) {
            self.error_message = Some(format!("Failed to set delay channels: {}", e));
            self.controls.set_vm_running(false);
            return;
        }

        let execution_tx = tx.clone();
        self.vm_execution_thread = Some(thread::spawn(move || {
            // Start a thread to handle real-time output
            let output_execution_tx = execution_tx.clone();
            let _output_thread = thread::spawn(move || {
                while let Ok(output) = output_rx.recv() {
                    if output_execution_tx
                        .send(ExecutionResult::Output(output))
                        .is_err()
                    {
                        break;
                    }
                }
            });

            // Start a thread to handle completion notifications
            let completion_execution_tx = execution_tx.clone();
            let _completion_thread = thread::spawn(move || {
                while let Ok(exit_code) = completion_rx.recv() {
                    if completion_execution_tx
                        .send(ExecutionResult::Completed(exit_code))
                        .is_err()
                    {
                        break;
                    }
                }
            });

            // Start a thread to handle delay requests from VM
            let delay_execution_tx = execution_tx.clone();
            let _delay_thread = thread::spawn(move || {
                while let Ok(delay_ms) = delay_request_rx.recv() {
                    if delay_execution_tx
                        .send(ExecutionResult::DelayRequest(delay_ms))
                        .is_err()
                    {
                        break;
                    }
                }
            });

            // Run the VM - completion will be handled by the callback mechanism
            match emulator_clone.run() {
                Err(VmError::ExecutionError(code)) => {
                    let _ = execution_tx.send(ExecutionResult::Error(format!(
                        "VM execution failed with error code: {}",
                        code
                    )));
                }
                Err(e) => {
                    let _ = execution_tx.send(ExecutionResult::Error(format!(
                        "VM execution failed: {}",
                        e
                    )));
                }
                Ok(_) => {
                    // Success case is now handled by the completion callback
                    // Don't send completion here - let the callback mechanism handle it
                }
            }
        }));
    }

    fn reset_vm(&mut self) {
        self.emulator.reset();
        self.console.clear();
        self.console.add_output("🔄 VM reset completed");
        self.error_message = None;
    }

    fn handle_file_dialog(&mut self) {
        if let Some(file) = FileDialog::new()
            .add_filter("Smollu Bytecode", &["smolbc"])
            .add_filter("All files", &["*"])
            .pick_file()
        {
            self.load_file(file);
        }
    }

    fn check_execution_status(&mut self) {
        let mut should_clear_receiver = false;
        let mut delay_requests = Vec::new();

        if let Some(ref receiver) = self.execution_receiver {
            while let Ok(result) = receiver.try_recv() {
                match result {
                    ExecutionResult::Output(output) => {
                        self.console.add_output(&output);
                    }
                    ExecutionResult::Completed(exit_code) => {
                        self.console.add_output(&format!(
                            "✅ VM execution completed with exit code: {}",
                            exit_code
                        ));
                        self.controls.set_vm_running(false);
                        if let Some(handle) = self.vm_execution_thread.take() {
                            let _ = handle.join();
                        }
                        should_clear_receiver = true;
                    }
                    ExecutionResult::Error(error) => {
                        self.console.add_output(&format!("❌ {}", error));
                        self.controls.set_vm_running(false);
                        if let Some(handle) = self.vm_execution_thread.take() {
                            let _ = handle.join();
                        }
                        should_clear_receiver = true;
                    }
                    ExecutionResult::DelayRequest(delay_ms) => {
                        self.console
                            .add_output(&format!("⏱️  Delay: {}ms", delay_ms));
                        delay_requests.push(delay_ms);
                    }
                }
            }
        }

        // Handle delay requests after releasing the receiver borrow
        for delay_ms in delay_requests {
            self.handle_delay_request(delay_ms);
        }

        if should_clear_receiver {
            self.execution_receiver = None;
        }

        // Update LED matrix with current state
        let led_matrix = self.emulator.get_led_matrix();
        self.led_matrix.update_matrix(led_matrix);
    }

    /// Handle a delay request from the VM - start GUI countdown
    fn handle_delay_request(&mut self, delay_ms: u32) {
        self.delay_start_time = Some(Instant::now());
        self.delay_duration = Duration::from_millis(delay_ms as u64);

        // Update LED matrix delay state for visualization
        let led_manager = get_led_matrix_manager();
        led_manager.with_current_matrix(|matrix| matrix.delay_ms(delay_ms));
    }

    /// Check if delay is complete and signal VM if needed
    fn check_delay_completion(&mut self) {
        if let Some(start_time) = self.delay_start_time {
            if start_time.elapsed() >= self.delay_duration {
                // Delay is complete - signal the VM
                if let Some(ref sender) = self.delay_completion_sender {
                    let _ = sender.send(());
                }

                // Clear delay state
                self.delay_start_time = None;
                self.delay_duration = Duration::from_millis(0);

                // Clear LED matrix delay state
                let led_manager = get_led_matrix_manager();
                led_manager.with_current_matrix(|matrix| matrix.clear_delay());

                self.console.add_output("✅ Delay completed");
            }
        }
    }
}

impl eframe::App for SmolluEmulatorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.check_execution_status();
        self.check_delay_completion();

        // LED Matrix side panel
        egui::SidePanel::right("led_matrix_panel")
            .default_width(400.0)
            .min_width(300.0)
            .show(ctx, |ui| {
                ui.heading("LED Matrix");
                ui.separator();
                self.led_matrix.show_with_controls(ui);
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading(RichText::new("Smollu VM Emulator").size(24.0).strong());
            ui.separator();

            if let Some(ref error) = self.error_message {
                ui.colored_label(Color32::RED, format!("❌ {}", error));
                ui.separator();
            }

            egui::TopBottomPanel::top("controls").show_inside(ui, |ui| {
                if let Some(action) = self.controls.show(ui) {
                    match action {
                        ControlAction::LoadFile => self.handle_file_dialog(),
                        ControlAction::Run => self.run_vm(),
                        ControlAction::Reset => self.reset_vm(),
                    }
                }
            });

            egui::CentralPanel::default().show_inside(ui, |ui| {
                self.console.show(ui);
            });
        });

        if self.vm_execution_thread.is_some() {
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {
        if let Some(handle) = self.vm_execution_thread.take() {
            let _ = handle.join();
        }
    }
}
