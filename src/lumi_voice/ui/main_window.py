"""LUMI Voice Assistant — PySide6 Desktop Interface.

Provides a complete GUI for testing and demonstrating the voice pipeline
with microphone/speaker selection, push-to-talk, continuous listening,
and conversation display.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Optional

from PySide6.QtCore import Qt, Signal, QObject, QTimer, Slot
from PySide6.QtGui import QFont, QTextCursor, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from lumi_voice.audio_devices import AudioDeviceManager
from lumi_voice.config import LumiConfig, load_config, save_user_config
from lumi_voice.logging_config import setup_logging
from lumi_voice.pipeline import ComponentStatus, PipelineState, VoicePipeline

logger = logging.getLogger(__name__)


class PipelineSignals(QObject):
    """Qt signals for thread-safe pipeline event delivery."""

    state_changed = Signal(str)
    transcript_ready = Signal(str)
    response_ready = Signal(str)
    response_token = Signal(str)
    error_occurred = Signal(str)
    audio_level = Signal(float)
    status_updated = Signal(object)
    init_complete = Signal(dict)


class MainWindow(QMainWindow):
    """Main application window for LUMI Voice Assistant."""

    def __init__(self, config: LumiConfig) -> None:
        super().__init__()
        self.config = config
        self.signals = PipelineSignals()
        self.device_manager = AudioDeviceManager()

        # Pipeline (initialized in background)
        self.pipeline: Optional[VoicePipeline] = None
        self._ptt_active = False
        self._continuous_active = False
        self._response_buffer = ""

        self._setup_ui()
        self._connect_signals()
        self._populate_devices()

        # Initialize pipeline in background
        QTimer.singleShot(100, self._init_pipeline_background)

    def _setup_ui(self) -> None:
        """Build the complete UI layout."""
        self.setWindowTitle("LUMI Voice Assistant")
        self.setMinimumSize(900, 700)
        self.resize(1050, 780)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(8)

        # --- Top: Status Bar ---
        status_group = QGroupBox("Component Status")
        status_layout = QHBoxLayout(status_group)
        self.status_labels: dict[str, QLabel] = {}
        for name in ["Microphone", "STT", "LLM", "TTS", "Speaker", "ROS 2"]:
            lbl = QLabel(f"{name}: ...")
            lbl.setStyleSheet("padding: 3px 8px; border-radius: 3px; background: #555; color: white;")
            lbl.setAlignment(Qt.AlignCenter)
            status_layout.addWidget(lbl)
            self.status_labels[name.lower().replace(" ", "")] = lbl
        main_layout.addWidget(status_group)

        # --- Middle: Tabs for interaction and settings ---
        tabs = QTabWidget()
        main_layout.addWidget(tabs, stretch=1)

        # Tab 1: Conversation
        conv_tab = QWidget()
        conv_layout = QVBoxLayout(conv_tab)
        tabs.addTab(conv_tab, "Conversation")

        # Conversation display
        self.conversation_display = QTextEdit()
        self.conversation_display.setReadOnly(True)
        self.conversation_display.setFont(QFont("Segoe UI", 11))
        self.conversation_display.setStyleSheet(
            "QTextEdit { background: #1e1e2e; color: #cdd6f4; border: 1px solid #45475a; border-radius: 6px; padding: 8px; }"
        )
        conv_layout.addWidget(self.conversation_display, stretch=1)

        # Pipeline state label
        self.state_label = QLabel("State: Idle")
        self.state_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.state_label.setAlignment(Qt.AlignCenter)
        conv_layout.addWidget(self.state_label)

        # Audio level bar
        self.level_bar = QProgressBar()
        self.level_bar.setRange(0, 100)
        self.level_bar.setValue(0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setMaximumHeight(8)
        self.level_bar.setStyleSheet(
            "QProgressBar { background: #313244; border-radius: 4px; }"
            "QProgressBar::chunk { background: #a6e3a1; border-radius: 4px; }"
        )
        conv_layout.addWidget(self.level_bar)

        # Controls row
        ctrl_layout = QHBoxLayout()
        conv_layout.addLayout(ctrl_layout)

        self.ptt_button = QPushButton("🎤 Push to Talk")
        self.ptt_button.setMinimumHeight(44)
        self.ptt_button.setStyleSheet(
            "QPushButton { background: #89b4fa; color: #1e1e2e; font-weight: bold; font-size: 13px; border-radius: 6px; padding: 8px 16px; }"
            "QPushButton:pressed { background: #f38ba8; }"
            "QPushButton:disabled { background: #585b70; color: #a6adc8; }"
        )
        ctrl_layout.addWidget(self.ptt_button)

        self.continuous_button = QPushButton("🔄 Continuous Listen")
        self.continuous_button.setCheckable(True)
        self.continuous_button.setMinimumHeight(44)
        self.continuous_button.setStyleSheet(
            "QPushButton { background: #a6e3a1; color: #1e1e2e; font-weight: bold; font-size: 13px; border-radius: 6px; padding: 8px 16px; }"
            "QPushButton:checked { background: #f38ba8; }"
            "QPushButton:disabled { background: #585b70; color: #a6adc8; }"
        )
        ctrl_layout.addWidget(self.continuous_button)

        self.clear_button = QPushButton("🗑 Clear Chat")
        self.clear_button.setMinimumHeight(44)
        ctrl_layout.addWidget(self.clear_button)

        # Text input row
        text_layout = QHBoxLayout()
        conv_layout.addLayout(text_layout)

        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type a message to LUMI (bypasses microphone)...")
        self.text_input.setMinimumHeight(36)
        text_layout.addWidget(self.text_input, stretch=1)

        self.send_button = QPushButton("Send")
        self.send_button.setMinimumHeight(36)
        text_layout.addWidget(self.send_button)

        # Tab 2: Audio Devices
        devices_tab = QWidget()
        devices_layout = QVBoxLayout(devices_tab)
        tabs.addTab(devices_tab, "Audio Devices")

        # Input devices
        input_group = QGroupBox("Microphone (Input)")
        input_layout = QVBoxLayout(input_group)
        self.input_combo = QComboBox()
        self.input_combo.setMinimumHeight(32)
        input_layout.addWidget(self.input_combo)
        input_btn_layout = QHBoxLayout()
        self.refresh_input_btn = QPushButton("🔄 Refresh Devices")
        self.test_mic_btn = QPushButton("🎤 Test Microphone")
        input_btn_layout.addWidget(self.refresh_input_btn)
        input_btn_layout.addWidget(self.test_mic_btn)
        input_layout.addLayout(input_btn_layout)
        self.mic_test_result = QLabel("")
        self.mic_test_result.setWordWrap(True)
        input_layout.addWidget(self.mic_test_result)
        devices_layout.addWidget(input_group)

        # Output devices
        output_group = QGroupBox("Speaker (Output)")
        output_layout = QVBoxLayout(output_group)
        self.output_combo = QComboBox()
        self.output_combo.setMinimumHeight(32)
        output_layout.addWidget(self.output_combo)
        output_btn_layout = QHBoxLayout()
        self.refresh_output_btn = QPushButton("🔄 Refresh Devices")
        self.test_speaker_btn = QPushButton("🔊 Test Speaker")
        output_btn_layout.addWidget(self.refresh_output_btn)
        output_btn_layout.addWidget(self.test_speaker_btn)
        output_layout.addLayout(output_btn_layout)
        self.speaker_test_result = QLabel("")
        self.speaker_test_result.setWordWrap(True)
        output_layout.addWidget(self.speaker_test_result)
        devices_layout.addWidget(output_group)
        devices_layout.addStretch()

        # Tab 3: Settings
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        tabs.addTab(settings_tab, "Settings")

        # STT settings
        stt_group = QGroupBox("Speech-to-Text (Whisper)")
        stt_layout = QVBoxLayout(stt_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Model Size:"))
        self.stt_model_combo = QComboBox()
        self.stt_model_combo.addItems(["tiny", "base", "small", "medium", "large-v3"])
        self.stt_model_combo.setCurrentText(self.config.stt.model_size)
        row.addWidget(self.stt_model_combo)
        stt_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Device:"))
        self.stt_device_combo = QComboBox()
        self.stt_device_combo.addItems(["cpu", "cuda"])
        self.stt_device_combo.setCurrentText(self.config.stt.device)
        row.addWidget(self.stt_device_combo)
        stt_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Language:"))
        self.stt_lang_input = QLineEdit(self.config.stt.language)
        self.stt_lang_input.setMaximumWidth(80)
        row.addWidget(self.stt_lang_input)
        row.addStretch()
        stt_layout.addLayout(row)

        settings_layout.addWidget(stt_group)

        # LLM settings
        llm_group = QGroupBox("Language Model (Ollama)")
        llm_layout = QVBoxLayout(llm_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Ollama Host:"))
        self.llm_host_input = QLineEdit(self.config.llm.ollama.host)
        row.addWidget(self.llm_host_input)
        llm_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Model:"))
        self.llm_model_input = QLineEdit(self.config.llm.ollama.model)
        row.addWidget(self.llm_model_input)
        llm_layout.addLayout(row)

        settings_layout.addWidget(llm_group)

        # TTS settings
        tts_group = QGroupBox("Text-to-Speech (Piper)")
        tts_layout = QVBoxLayout(tts_group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Voice Model:"))
        self.tts_voice_input = QLineEdit(self.config.tts.voice_model)
        row.addWidget(self.tts_voice_input)
        tts_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Speed (length_scale):"))
        self.tts_speed_input = QLineEdit(str(self.config.tts.length_scale))
        self.tts_speed_input.setMaximumWidth(80)
        row.addWidget(self.tts_speed_input)
        row.addStretch()
        tts_layout.addLayout(row)

        settings_layout.addWidget(tts_group)

        # Save settings button
        self.save_settings_btn = QPushButton("💾 Save Settings")
        self.save_settings_btn.setMinimumHeight(36)
        settings_layout.addWidget(self.save_settings_btn)
        settings_layout.addStretch()

        # Tab 4: Logs
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        tabs.addTab(log_tab, "Logs")

        self.log_display = QPlainTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 9))
        self.log_display.setMaximumBlockCount(1000)
        self.log_display.setStyleSheet(
            "QPlainTextEdit { background: #11111b; color: #a6adc8; border: 1px solid #45475a; }"
        )
        log_layout.addWidget(self.log_display)

        # Install log handler
        self._log_handler = QtLogHandler(self.log_display)
        logging.getLogger().addHandler(self._log_handler)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Initializing LUMI Voice Assistant...")

    def _connect_signals(self) -> None:
        """Connect UI signals to slots."""
        # Pipeline signals
        self.signals.state_changed.connect(self._on_state_changed)
        self.signals.transcript_ready.connect(self._on_transcript)
        self.signals.response_ready.connect(self._on_response)
        self.signals.response_token.connect(self._on_response_token)
        self.signals.error_occurred.connect(self._on_error)
        self.signals.audio_level.connect(self._on_level)
        self.signals.status_updated.connect(self._on_status_updated)
        self.signals.init_complete.connect(self._on_init_complete)

        # Button actions
        self.ptt_button.pressed.connect(self._ptt_pressed)
        self.ptt_button.released.connect(self._ptt_released)
        self.continuous_button.toggled.connect(self._continuous_toggled)
        self.clear_button.clicked.connect(self._clear_conversation)
        self.send_button.clicked.connect(self._send_text)
        self.text_input.returnPressed.connect(self._send_text)

        # Device controls
        self.refresh_input_btn.clicked.connect(self._refresh_devices)
        self.refresh_output_btn.clicked.connect(self._refresh_devices)
        self.test_mic_btn.clicked.connect(self._test_microphone)
        self.test_speaker_btn.clicked.connect(self._test_speaker)
        self.input_combo.currentIndexChanged.connect(self._input_device_changed)
        self.output_combo.currentIndexChanged.connect(self._output_device_changed)

        # Settings
        self.save_settings_btn.clicked.connect(self._save_settings)

    def _populate_devices(self) -> None:
        """Populate device combo boxes."""
        self.input_combo.blockSignals(True)
        self.output_combo.blockSignals(True)

        self.input_combo.clear()
        self.output_combo.clear()

        self.input_combo.addItem("System Default", None)
        for dev in self.device_manager.input_devices:
            self.input_combo.addItem(dev.display_name(), dev.index)

        self.output_combo.addItem("System Default", None)
        for dev in self.device_manager.output_devices:
            self.output_combo.addItem(dev.display_name(), dev.index)

        # Restore saved selection
        if self.config.audio_input.device_index is not None:
            for i in range(self.input_combo.count()):
                if self.input_combo.itemData(i) == self.config.audio_input.device_index:
                    self.input_combo.setCurrentIndex(i)
                    break

        if self.config.audio_output.device_index is not None:
            for i in range(self.output_combo.count()):
                if self.output_combo.itemData(i) == self.config.audio_output.device_index:
                    self.output_combo.setCurrentIndex(i)
                    break

        self.input_combo.blockSignals(False)
        self.output_combo.blockSignals(False)

    def _init_pipeline_background(self) -> None:
        """Initialize the pipeline in a background thread."""
        thread = threading.Thread(target=self._init_pipeline, daemon=True)
        thread.start()

    def _init_pipeline(self) -> None:
        """Initialize pipeline components (runs in background)."""
        try:
            self.pipeline = VoicePipeline(
                config=self.config,
                on_state_change=lambda s: self.signals.state_changed.emit(s.value),
                on_transcript=lambda t: self.signals.transcript_ready.emit(t),
                on_response=lambda r: self.signals.response_ready.emit(r),
                on_response_token=lambda t: self.signals.response_token.emit(t),
                on_error=lambda e: self.signals.error_occurred.emit(e),
                on_level=lambda l: self.signals.audio_level.emit(l),
                on_status_change=lambda s: self.signals.status_updated.emit(s),
            )
            results = self.pipeline.initialize()
            self.signals.init_complete.emit(results)
        except Exception as e:
            self.signals.error_occurred.emit(f"Pipeline initialization failed: {e}")

    # --- Signal handlers ---

    @Slot(str)
    def _on_state_changed(self, state: str) -> None:
        state_display = {
            "idle": "⏸ Idle",
            "listening": "🎤 Listening...",
            "transcribing": "📝 Transcribing...",
            "thinking": "🧠 Thinking...",
            "speaking": "🔊 Speaking...",
            "error": "❌ Error",
        }
        self.state_label.setText(f"State: {state_display.get(state, state)}")
        self.status_bar.showMessage(f"Pipeline: {state}")

    @Slot(str)
    def _on_transcript(self, text: str) -> None:
        self.conversation_display.append(
            f'<div style="margin: 6px 0;"><b style="color: #89b4fa;">You:</b> '
            f'<span style="color: #cdd6f4;">{text}</span></div>'
        )
        self._response_buffer = ""

    @Slot(str)
    def _on_response(self, text: str) -> None:
        # Full response received - if we were streaming tokens, this replaces
        if self._response_buffer:
            # Already streamed tokens, response is complete
            self._response_buffer = ""
        else:
            self.conversation_display.append(
                f'<div style="margin: 6px 0;"><b style="color: #a6e3a1;">LUMI:</b> '
                f'<span style="color: #cdd6f4;">{text}</span></div>'
            )

    @Slot(str)
    def _on_response_token(self, token: str) -> None:
        if not self._response_buffer:
            # Start of a new response
            self.conversation_display.append(
                f'<div style="margin: 6px 0;"><b style="color: #a6e3a1;">LUMI:</b> '
                f'<span style="color: #cdd6f4;">'
            )
        self._response_buffer += token
        cursor = self.conversation_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(token)
        self.conversation_display.setTextCursor(cursor)
        self.conversation_display.ensureCursorVisible()

    @Slot(str)
    def _on_error(self, msg: str) -> None:
        self.conversation_display.append(
            f'<div style="margin: 6px 0; color: #f38ba8;"><b>⚠ Error:</b> {msg}</div>'
        )
        self.status_bar.showMessage(f"Error: {msg}", 10000)

    @Slot(float)
    def _on_level(self, level: float) -> None:
        self.level_bar.setValue(int(min(level * 500, 100)))  # Scale for visibility

    @Slot(object)
    def _on_status_updated(self, status: ComponentStatus) -> None:
        status_colors = {
            "ready": "#a6e3a1",
            "loaded": "#a6e3a1",
            "connected": "#a6e3a1",
            "unavailable": "#f38ba8",
            "not configured": "#fab387",
            "disconnected": "#f38ba8",
        }

        mapping = {
            "microphone": status.microphone,
            "stt": status.stt,
            "llm": status.llm,
            "tts": status.tts,
            "speaker": status.speaker,
            "ros2": status.ros2,
        }

        for key, state in mapping.items():
            if key in self.status_labels:
                color = status_colors.get(state, "#fab387")
                display = key.upper() if key in ("stt", "llm", "tts", "ros2") else key.capitalize()
                self.status_labels[key].setText(f"{display}: {state}")
                self.status_labels[key].setStyleSheet(
                    f"padding: 3px 8px; border-radius: 3px; background: {color}; color: #1e1e2e; font-weight: bold;"
                )

    @Slot(dict)
    def _on_init_complete(self, results: dict) -> None:
        for component, (success, msg) in results.items():
            level = "✅" if success else "⚠️"
            self.conversation_display.append(
                f'<div style="color: {"#a6e3a1" if success else "#fab387"};">'
                f"{level} {component.upper()}: {msg}</div>"
            )
        self.status_bar.showMessage("LUMI Voice Assistant ready")
        self.ptt_button.setEnabled(True)
        self.continuous_button.setEnabled(True)

    # --- Control handlers ---

    def _ptt_pressed(self) -> None:
        if self.pipeline and not self._continuous_active:
            self._ptt_active = True
            self.pipeline.start_ptt_recording()

    def _ptt_released(self) -> None:
        if self.pipeline and self._ptt_active:
            self._ptt_active = False
            self.pipeline.stop_ptt_recording()

    def _continuous_toggled(self, checked: bool) -> None:
        if not self.pipeline:
            return
        if checked:
            self._continuous_active = True
            self.continuous_button.setText("⏹ Stop Listening")
            self.ptt_button.setEnabled(False)
            self.pipeline.start_continuous_listening()
        else:
            self._continuous_active = False
            self.continuous_button.setText("🔄 Continuous Listen")
            self.ptt_button.setEnabled(True)
            self.pipeline.stop_continuous_listening()

    def _clear_conversation(self) -> None:
        self.conversation_display.clear()
        if self.pipeline:
            self.pipeline.clear_conversation()
        self.conversation_display.append(
            '<div style="color: #6c7086;"><i>Conversation cleared.</i></div>'
        )

    def _send_text(self) -> None:
        text = self.text_input.text().strip()
        if text and self.pipeline:
            self.text_input.clear()
            self.pipeline.process_text_input(text)

    # --- Device handlers ---

    def _refresh_devices(self) -> None:
        self.device_manager.refresh()
        self._populate_devices()
        self.status_bar.showMessage("Audio devices refreshed", 3000)

    def _input_device_changed(self, index: int) -> None:
        device_index = self.input_combo.itemData(index)
        if self.pipeline:
            self.pipeline.set_input_device(device_index)
        self.config.audio_input.device_index = device_index

    def _output_device_changed(self, index: int) -> None:
        device_index = self.output_combo.itemData(index)
        if self.pipeline:
            self.pipeline.set_output_device(device_index)
        self.config.audio_output.device_index = device_index

    def _test_microphone(self) -> None:
        self.mic_test_result.setText("Testing microphone...")
        device_index = self.input_combo.currentData()
        thread = threading.Thread(
            target=self._run_mic_test, args=(device_index,), daemon=True
        )
        thread.start()

    def _run_mic_test(self, device_index: Optional[int]) -> None:
        success, msg = self.device_manager.test_input_device(device_index)
        self.signals.error_occurred.emit(f"Mic test: {msg}" if not success else "")
        # Use a timer to update UI from main thread
        QTimer.singleShot(0, lambda: self.mic_test_result.setText(f"{'✅' if success else '❌'} {msg}"))

    def _test_speaker(self) -> None:
        self.speaker_test_result.setText("Testing speaker...")
        device_index = self.output_combo.currentData()
        thread = threading.Thread(
            target=self._run_speaker_test, args=(device_index,), daemon=True
        )
        thread.start()

    def _run_speaker_test(self, device_index: Optional[int]) -> None:
        success, msg = self.device_manager.test_output_device(device_index)
        QTimer.singleShot(0, lambda: self.speaker_test_result.setText(f"{'✅' if success else '❌'} {msg}"))

    def _save_settings(self) -> None:
        """Save current settings to user config."""
        self.config.stt.model_size = self.stt_model_combo.currentText()
        self.config.stt.device = self.stt_device_combo.currentText()
        self.config.stt.language = self.stt_lang_input.text().strip() or "en"
        self.config.llm.ollama.host = self.llm_host_input.text().strip()
        self.config.llm.ollama.model = self.llm_model_input.text().strip()
        self.config.tts.voice_model = self.tts_voice_input.text().strip()
        try:
            self.config.tts.length_scale = float(self.tts_speed_input.text())
        except ValueError:
            pass

        self.config.audio_input.device_index = self.input_combo.currentData()
        self.config.audio_output.device_index = self.output_combo.currentData()

        try:
            save_user_config(self.config)
            self.status_bar.showMessage("Settings saved to config/user.yaml", 5000)
        except Exception as e:
            self.status_bar.showMessage(f"Failed to save settings: {e}", 5000)

    def closeEvent(self, event) -> None:
        """Clean shutdown on window close."""
        if self.pipeline:
            self.pipeline.shutdown()
        event.accept()


class QtLogHandler(logging.Handler):
    """Sends log records to a QPlainTextEdit widget."""

    def __init__(self, text_widget: QPlainTextEdit) -> None:
        super().__init__()
        self.text_widget = text_widget
        self.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            # Thread-safe append
            QTimer.singleShot(0, lambda m=msg: self.text_widget.appendPlainText(m))
        except Exception:
            pass


def launch_ui(config: LumiConfig | None = None) -> int:
    """Launch the LUMI Voice Assistant GUI."""
    if config is None:
        config = load_config()

    setup_logging(level=config.logging.level, log_file=config.logging.file)

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("LUMI Voice Assistant")
    app.setStyle("Fusion")

    # Dark theme
    from PySide6.QtGui import QPalette, QColor
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 46))
    palette.setColor(QPalette.WindowText, QColor(205, 214, 244))
    palette.setColor(QPalette.Base, QColor(24, 24, 37))
    palette.setColor(QPalette.AlternateBase, QColor(30, 30, 46))
    palette.setColor(QPalette.ToolTipBase, QColor(49, 50, 68))
    palette.setColor(QPalette.ToolTipText, QColor(205, 214, 244))
    palette.setColor(QPalette.Text, QColor(205, 214, 244))
    palette.setColor(QPalette.Button, QColor(49, 50, 68))
    palette.setColor(QPalette.ButtonText, QColor(205, 214, 244))
    palette.setColor(QPalette.BrightText, QColor(243, 139, 168))
    palette.setColor(QPalette.Highlight, QColor(137, 180, 250))
    palette.setColor(QPalette.HighlightedText, QColor(30, 30, 46))
    app.setPalette(palette)

    window = MainWindow(config)
    window.show()
    return app.exec()
