"""Audio device enumeration and management."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import sounddevice as sd

logger = logging.getLogger(__name__)


@dataclass
class AudioDevice:
    """Represents an audio device."""
    index: int
    name: str
    max_input_channels: int = 0
    max_output_channels: int = 0
    default_sample_rate: float = 0.0
    host_api: str = ""
    is_default_input: bool = False
    is_default_output: bool = False

    @property
    def is_input(self) -> bool:
        return self.max_input_channels > 0

    @property
    def is_output(self) -> bool:
        return self.max_output_channels > 0

    def display_name(self) -> str:
        tags = []
        if self.is_default_input:
            tags.append("Default Input")
        if self.is_default_output:
            tags.append("Default Output")
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        return f"[{self.index}] {self.name}{tag_str}"


class AudioDeviceManager:
    """Manages enumeration and selection of audio input/output devices."""

    def __init__(self) -> None:
        self._input_devices: list[AudioDevice] = []
        self._output_devices: list[AudioDevice] = []
        self._selected_input: Optional[AudioDevice] = None
        self._selected_output: Optional[AudioDevice] = None
        self.refresh()

    def refresh(self) -> None:
        """Re-enumerate all available audio devices."""
        self._input_devices.clear()
        self._output_devices.clear()

        try:
            devices = sd.query_devices()
            host_apis = sd.query_hostapis()

            default_input_idx = sd.default.device[0]
            default_output_idx = sd.default.device[1]
            # sd.default.device can return a single int or a list
            if isinstance(default_input_idx, (list, tuple)):
                default_input_idx = default_input_idx[0]
            if isinstance(default_output_idx, (list, tuple)):
                default_output_idx = default_output_idx[0]

        except Exception as e:
            logger.error("Failed to enumerate audio devices: %s", e)
            return

        if not isinstance(devices, list):
            devices = [devices]

        for i, dev in enumerate(devices):
            if not isinstance(dev, dict):
                continue
            api_name = ""
            api_idx = dev.get("hostapi", -1)
            if isinstance(api_idx, int) and 0 <= api_idx < len(host_apis):
                api_name = host_apis[api_idx].get("name", "")

            device = AudioDevice(
                index=i,
                name=dev.get("name", f"Device {i}"),
                max_input_channels=dev.get("max_input_channels", 0),
                max_output_channels=dev.get("max_output_channels", 0),
                default_sample_rate=dev.get("default_samplerate", 0.0),
                host_api=api_name,
                is_default_input=(i == default_input_idx),
                is_default_output=(i == default_output_idx),
            )

            if device.is_input:
                self._input_devices.append(device)
            if device.is_output:
                self._output_devices.append(device)

        logger.info(
            "Found %d input devices, %d output devices",
            len(self._input_devices),
            len(self._output_devices),
        )

    @property
    def input_devices(self) -> list[AudioDevice]:
        return list(self._input_devices)

    @property
    def output_devices(self) -> list[AudioDevice]:
        return list(self._output_devices)

    @property
    def selected_input(self) -> Optional[AudioDevice]:
        return self._selected_input

    @property
    def selected_output(self) -> Optional[AudioDevice]:
        return self._selected_output

    def select_input(self, device_index: Optional[int]) -> Optional[AudioDevice]:
        """Select an input device by index. None = system default."""
        if device_index is None:
            self._selected_input = None
            logger.info("Input device set to system default")
            return None

        for dev in self._input_devices:
            if dev.index == device_index:
                self._selected_input = dev
                logger.info("Selected input device: %s", dev.display_name())
                return dev

        logger.warning("Input device index %d not found", device_index)
        return None

    def select_output(self, device_index: Optional[int]) -> Optional[AudioDevice]:
        """Select an output device by index. None = system default."""
        if device_index is None:
            self._selected_output = None
            logger.info("Output device set to system default")
            return None

        for dev in self._output_devices:
            if dev.index == device_index:
                self._selected_output = dev
                logger.info("Selected output device: %s", dev.display_name())
                return dev

        logger.warning("Output device index %d not found", device_index)
        return None

    def get_input_device_index(self) -> Optional[int]:
        """Get the currently selected input device index (None = default)."""
        return self._selected_input.index if self._selected_input else None

    def get_output_device_index(self) -> Optional[int]:
        """Get the currently selected output device index (None = default)."""
        return self._selected_output.index if self._selected_output else None

    def is_device_available(self, device_index: int) -> bool:
        """Check if a device index is currently available."""
        all_devices = self._input_devices + self._output_devices
        return any(d.index == device_index for d in all_devices)

    def test_input_device(self, device_index: Optional[int] = None, duration_s: float = 1.0) -> tuple[bool, str]:
        """Test an input device by recording a short clip. Returns (success, message)."""
        idx = device_index if device_index is not None else self.get_input_device_index()
        try:
            import numpy as np
            recording = sd.rec(
                int(16000 * duration_s),
                samplerate=16000,
                channels=1,
                device=idx,
                dtype="float32",
            )
            sd.wait()
            peak = float(np.max(np.abs(recording)))
            if peak < 0.001:
                return True, f"Recording captured but very quiet (peak={peak:.4f}). Check microphone."
            return True, f"Microphone working. Peak level: {peak:.4f}"
        except Exception as e:
            return False, f"Microphone test failed: {e}"

    def test_output_device(self, device_index: Optional[int] = None, duration_s: float = 0.5) -> tuple[bool, str]:
        """Test an output device by playing a short tone. Returns (success, message)."""
        idx = device_index if device_index is not None else self.get_output_device_index()
        try:
            import numpy as np
            sample_rate = 44100
            t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
            tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32")
            sd.play(tone, samplerate=sample_rate, device=idx)
            sd.wait()
            return True, "Test tone played successfully."
        except Exception as e:
            return False, f"Speaker test failed: {e}"
