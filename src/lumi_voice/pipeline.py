"""Voice pipeline — orchestrates STT → LLM → TTS flow."""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Callable, Optional

import numpy as np

from lumi_voice.audio_capture import AudioCapture
from lumi_voice.audio_playback import AudioPlayer
from lumi_voice.config import LumiConfig
from lumi_voice.conversation import ConversationManager
from lumi_voice.llm import LLMEngine
from lumi_voice.stt import SpeechToText
from lumi_voice.tts import TextToSpeech
from lumi_voice.vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


class ComponentStatus:
    """Status tracker for all pipeline components."""

    def __init__(self) -> None:
        self.microphone: str = "unavailable"
        self.stt: str = "unavailable"
        self.llm: str = "unavailable"
        self.tts: str = "unavailable"
        self.speaker: str = "unavailable"
        self.ros2: str = "not configured"

    def to_dict(self) -> dict[str, str]:
        return {
            "microphone": self.microphone,
            "stt": self.stt,
            "llm": self.llm,
            "tts": self.tts,
            "speaker": self.speaker,
            "ros2": self.ros2,
        }


class VoicePipeline:
    """Orchestrates the complete voice interaction pipeline."""

    def __init__(
        self,
        config: LumiConfig,
        on_state_change: Optional[Callable[[PipelineState], None]] = None,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_response: Optional[Callable[[str], None]] = None,
        on_response_token: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_level: Optional[Callable[[float], None]] = None,
        on_status_change: Optional[Callable[[ComponentStatus], None]] = None,
    ) -> None:
        self.config = config
        self._state = PipelineState.IDLE
        self.status = ComponentStatus()

        # Callbacks
        self._on_state_change = on_state_change
        self._on_transcript = on_transcript
        self._on_response = on_response
        self._on_response_token = on_response_token
        self._on_error = on_error
        self._on_level = on_level
        self._on_status_change = on_status_change

        # Components
        self._capture = AudioCapture(
            sample_rate=config.audio_input.sample_rate,
            channels=config.audio_input.channels,
            chunk_duration_ms=config.audio_input.chunk_duration_ms,
            device_index=config.audio_input.device_index,
            dtype=config.audio_input.dtype,
        )
        self._player = AudioPlayer(device_index=config.audio_output.device_index)
        self._vad = VoiceActivityDetector(
            energy_threshold=config.vad.energy_threshold,
            silence_duration_s=config.vad.silence_duration_s,
            min_speech_duration_s=config.vad.min_speech_duration_s,
            pre_speech_padding_s=config.vad.pre_speech_padding_s,
            sample_rate=config.audio_input.sample_rate,
            chunk_duration_ms=config.audio_input.chunk_duration_ms,
        )
        self._stt = SpeechToText(config.stt)
        self._llm = LLMEngine(config.llm)
        self._tts = TextToSpeech(config.tts)
        self._conversation = ConversationManager(
            system_prompt=config.llm.conversation.system_prompt,
            max_history=config.llm.conversation.max_history,
            save_history=config.privacy.save_conversation_history,
            history_path=config.privacy.conversation_history_path,
        )

        # Threading
        self._listen_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._ptt_recording = False
        self._ptt_chunks: list[np.ndarray] = []

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def conversation(self) -> ConversationManager:
        return self._conversation

    def _set_state(self, state: PipelineState) -> None:
        self._state = state
        if self._on_state_change:
            try:
                self._on_state_change(state)
            except Exception:
                pass

    def _emit_error(self, msg: str) -> None:
        logger.error(msg)
        if self._on_error:
            try:
                self._on_error(msg)
            except Exception:
                pass

    def _update_status(self) -> None:
        if self._on_status_change:
            try:
                self._on_status_change(self.status)
            except Exception:
                pass

    def initialize(self) -> dict[str, tuple[bool, str]]:
        """Initialize all components. Returns status dict."""
        results: dict[str, tuple[bool, str]] = {}

        # STT
        try:
            self._stt.load_model()
            self.status.stt = "loaded"
            results["stt"] = (True, f"Whisper model '{self.config.stt.model_size}' loaded")
        except Exception as e:
            self.status.stt = "unavailable"
            results["stt"] = (False, str(e))

        # LLM
        avail, msg = self._llm.check_availability()
        self.status.llm = "connected" if avail else "unavailable"
        results["llm"] = (avail, msg)

        # TTS
        avail, msg = self._tts.check_availability()
        self.status.tts = "ready" if avail else "unavailable"
        results["tts"] = (avail, msg)

        # Mic and Speaker default to ready (tested on demand)
        self.status.microphone = "ready"
        self.status.speaker = "ready"
        results["microphone"] = (True, "Microphone available")
        results["speaker"] = (True, "Speaker available")

        # ROS 2
        self.status.ros2 = "not configured"
        results["ros2"] = (False, "ROS 2 bridge not configured")

        self._update_status()
        return results

    # --- Push-to-Talk ---

    def start_ptt_recording(self) -> None:
        """Start recording for push-to-talk."""
        self._ptt_chunks.clear()
        self._ptt_recording = True
        if self._on_level:
            self._capture.set_level_callback(self._on_level)
        self._capture.start()
        self._set_state(PipelineState.LISTENING)
        logger.info("PTT recording started")

    def stop_ptt_recording(self) -> None:
        """Stop recording and process the captured audio."""
        if not self._ptt_recording:
            return

        self._ptt_recording = False
        chunks = self._capture.read_all_chunks()
        self._capture.stop()

        if not chunks:
            logger.warning("No audio captured during PTT")
            self._set_state(PipelineState.IDLE)
            return

        audio = np.concatenate(chunks, axis=0)
        logger.info("PTT captured %d samples (%.2fs)", len(audio), len(audio) / self.config.audio_input.sample_rate)

        # Process in background thread
        thread = threading.Thread(target=self._process_audio, args=(audio,), daemon=True)
        thread.start()

    # --- Continuous Listening ---

    def start_continuous_listening(self) -> None:
        """Start continuous listening with VAD."""
        if self._listen_thread and self._listen_thread.is_alive():
            logger.warning("Continuous listening already active")
            return

        self._stop_event.clear()
        self._vad.reset()
        if self._on_level:
            self._capture.set_level_callback(self._on_level)
        self._capture.start()
        self._set_state(PipelineState.LISTENING)

        self._listen_thread = threading.Thread(
            target=self._continuous_listen_loop, daemon=True
        )
        self._listen_thread.start()
        logger.info("Continuous listening started")

    def stop_continuous_listening(self) -> None:
        """Stop continuous listening."""
        self._stop_event.set()
        self._capture.stop()
        if self._listen_thread:
            self._listen_thread.join(timeout=3)
        self._set_state(PipelineState.IDLE)
        logger.info("Continuous listening stopped")

    def _continuous_listen_loop(self) -> None:
        """Background loop for continuous listening."""
        while not self._stop_event.is_set():
            chunk = self._capture.read_chunk(timeout=0.5)
            if chunk is None:
                continue

            # Skip VAD processing while pipeline is busy
            if self._state not in (PipelineState.LISTENING,):
                continue

            utterance = self._vad.process_chunk(chunk)
            if utterance is not None:
                self._process_audio(utterance)
                # Wait for pipeline to finish before listening again
                self._set_state(PipelineState.LISTENING)
                self._vad.reset()

    # --- Core Processing ---

    def _process_audio(self, audio: np.ndarray) -> None:
        """Process captured audio through STT → LLM → TTS."""
        # Transcribe
        self._set_state(PipelineState.TRANSCRIBING)
        transcript = self._stt.transcribe_audio(audio)

        if not transcript:
            logger.info("No valid transcript from audio")
            self._set_state(PipelineState.IDLE if not self._listen_thread else PipelineState.LISTENING)
            return

        logger.info("User said: %s", transcript)
        if self._on_transcript:
            try:
                self._on_transcript(transcript)
            except Exception:
                pass

        # Generate response
        self._set_state(PipelineState.THINKING)
        self._conversation.add_user_message(transcript)
        messages = self._conversation.get_messages_for_api()

        full_response = ""
        if self.config.llm.ollama.stream and self._on_response_token:
            # Stream tokens to UI
            for token in self._llm.generate_stream(messages):
                full_response += token
                try:
                    self._on_response_token(token)
                except Exception:
                    pass
        else:
            full_response = self._llm.generate(messages)

        if not full_response:
            self._emit_error("LLM returned empty response")
            self._set_state(PipelineState.IDLE)
            return

        self._conversation.add_assistant_message(full_response)
        logger.info("LUMI response: %s", full_response[:200])
        if self._on_response:
            try:
                self._on_response(full_response)
            except Exception:
                pass

        # Synthesize and play
        self._set_state(PipelineState.SPEAKING)
        wav_data = self._tts.synthesize(full_response)
        if wav_data:
            self._player.play_wav_bytes(wav_data)
        else:
            logger.warning("TTS synthesis failed, skipping audio playback")

        self._set_state(PipelineState.IDLE if not self._listen_thread else PipelineState.LISTENING)

    # --- Controls ---

    def process_text_input(self, text: str) -> None:
        """Process direct text input (bypass STT)."""
        if not text.strip():
            return

        thread = threading.Thread(
            target=self._process_text, args=(text.strip(),), daemon=True
        )
        thread.start()

    def _process_text(self, text: str) -> None:
        """Process text through LLM → TTS."""
        self._set_state(PipelineState.THINKING)
        if self._on_transcript:
            try:
                self._on_transcript(text)
            except Exception:
                pass

        self._conversation.add_user_message(text)
        messages = self._conversation.get_messages_for_api()

        full_response = ""
        if self.config.llm.ollama.stream and self._on_response_token:
            for token in self._llm.generate_stream(messages):
                full_response += token
                try:
                    self._on_response_token(token)
                except Exception:
                    pass
        else:
            full_response = self._llm.generate(messages)

        if full_response:
            self._conversation.add_assistant_message(full_response)
            if self._on_response:
                try:
                    self._on_response(full_response)
                except Exception:
                    pass

            self._set_state(PipelineState.SPEAKING)
            wav_data = self._tts.synthesize(full_response)
            if wav_data:
                self._player.play_wav_bytes(wav_data)

        self._set_state(PipelineState.IDLE)

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        self._conversation.clear()

    def stop_all(self) -> None:
        """Stop all pipeline activity."""
        self._stop_event.set()
        self._capture.stop()
        self._player.stop()
        self._set_state(PipelineState.IDLE)

    def set_input_device(self, device_index: Optional[int]) -> None:
        """Change input device."""
        self._capture.set_device(device_index)
        self.config.audio_input.device_index = device_index

    def set_output_device(self, device_index: Optional[int]) -> None:
        """Change output device."""
        self._player.set_device(device_index)
        self.config.audio_output.device_index = device_index

    def shutdown(self) -> None:
        """Clean shutdown of all components."""
        logger.info("Shutting down voice pipeline")
        self.stop_all()
        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=3)
        self._stt.unload_model()
        if self.config.privacy.save_conversation_history:
            self._conversation.save_to_file()
        logger.info("Pipeline shutdown complete")
