"""
Camera device discovery and hardware portability utilities for LUMI Vision.

Enumerates available video capture devices across Windows (DirectShow/MSMF)
and Linux/Raspberry Pi (V4L2 /dev/video*), providing clear diagnostic reporting
for switching between laptop webcams and robot external cameras.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2

logger = logging.getLogger(__name__)


@dataclass
class CameraDeviceInfo:
    """Diagnostic details for an enumerated camera capture device."""
    index: Union[int, str]
    backend_name: str
    is_available: bool
    default_width: int = 0
    default_height: int = 0
    default_fps: float = 0.0
    notes: str = ""


def list_available_cameras(max_test_index: int = 6) -> List[CameraDeviceInfo]:
    """
    Probe candidate camera indices and return list of working camera devices.

    Args:
        max_test_index: Highest camera index to probe (default 0 through 5).

    Returns:
        List of CameraDeviceInfo for all probed devices.
    """
    devices: List[CameraDeviceInfo] = []

    # Choose primary backend based on operating system
    if sys.platform.startswith("win"):
        backends = [(cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "Media Foundation")]
    elif sys.platform.startswith("linux"):
        backends = [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_ANY, "Default")]
    else:
        backends = [(cv2.CAP_ANY, "Default")]

    for idx in range(max_test_index):
        opened = False
        cap = None
        used_backend = "Unknown"

        for b_id, b_name in backends:
            try:
                cap = cv2.VideoCapture(idx, b_id)
                if cap.isOpened():
                    opened = True
                    used_backend = b_name
                    break
                else:
                    cap.release()
            except Exception:
                if cap is not None:
                    cap.release()

        if opened and cap is not None:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            # Test grabbing a single frame to verify real stream viability
            ret, _ = cap.read()
            cap.release()

            status_note = "Ready (Frame capture verified)" if ret else "Opened (No frame received)"
            devices.append(
                CameraDeviceInfo(
                    index=idx,
                    backend_name=used_backend,
                    is_available=ret,
                    default_width=w,
                    default_height=h,
                    default_fps=fps,
                    notes=status_note,
                )
            )
        else:
            # Device not present at this index
            pass

    # On Linux / Raspberry Pi, also check device nodes in /dev/video*
    if sys.platform.startswith("linux"):
        for video_node in sorted(Path("/dev").glob("video*")):
            node_path = str(video_node)
            try:
                cap = cv2.VideoCapture(node_path, cv2.CAP_V4L2)
                if cap.isOpened():
                    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    ret, _ = cap.read()
                    cap.release()
                    devices.append(
                        CameraDeviceInfo(
                            index=node_path,
                            backend_name="V4L2-Node",
                            is_available=ret,
                            default_width=w,
                            default_height=h,
                            default_fps=fps,
                            notes="Linux Device Node",
                        )
                    )
            except Exception:
                pass

    return devices


def print_camera_diagnostics() -> None:
    """Print formatted camera device enumeration table to stdout."""
    print("\n========================================================")
    print("           LUMI Vision - Camera Device Discovery        ")
    print("========================================================")
    print("Scanning available video capture devices on system...\n")

    cams = list_available_cameras()
    if not cams:
        print("  [WARN] No usable camera devices detected!")
        print("  Suggestions:")
        print("    - Verify webcam is connected and unblocked by privacy shutter.")
        print("    - Ensure no other application (Zoom, Teams, browser) is locking the camera.")
        print("    - On Linux/Raspberry Pi, verify user is in 'video' group: sudo usermod -a -G video $USER")
        print("    - Run tests with synthetic camera: '--mock-camera'")
    else:
        print(f"  Found {len(cams)} accessible camera device(s):\n")
        print(f"  {'Index/Path':<12} | {'Backend':<14} | {'Resolution':<12} | {'FPS':<6} | {'Status'}")
        print("  " + "-" * 66)
        for cam in cams:
            res_str = f"{cam.default_width}x{cam.default_height}"
            idx_str = str(cam.index)
            print(f"  {idx_str:<12} | {cam.backend_name:<14} | {res_str:<12} | {cam.default_fps:<6.1f} | {cam.notes}")
    print("========================================================\n")
