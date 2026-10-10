#!/usr/bin/env python3
"""Scan USB bus and video devices for sensors OpenMotion can use.

Cross-platform:
  Linux  — sysfs USB enumeration + V4L2 device scan
  Windows — WMI Win32_PnPEntity USB enumeration + DirectShow webcam listing
  Other  — returns an empty list
"""

import os
import glob
import subprocess
import sys
from dataclasses import dataclass


LEAP_MOTION_VID = "f182"
LEAP_MOTION_VID_INT = 0xF182


@dataclass
class DetectedSensor:
    name: str
    type: str          # "leap_motion", "webcam"
    device_path: str
    usb_vid: str = ""
    usb_pid: str = ""


# ---------------------------------------------------------------------------
# Linux helpers
# ---------------------------------------------------------------------------

def _read_sysfs(path: str) -> str:
    """Read a single sysfs attribute, stripping whitespace."""
    with open(path) as f:
        return f.read().strip()


def _is_leap_v4l2(sysfs_path: str) -> bool:
    """Walk the real sysfs path upward to check if the V4L2 node
    belongs to a Leap Motion USB device."""
    real = os.path.realpath(sysfs_path)
    parts = real.split("/")
    for i in range(len(parts)):
        vid_path = "/".join(parts[: i + 1]) + "/idVendor"
        if os.path.exists(vid_path):
            try:
                if _read_sysfs(vid_path) == LEAP_MOTION_VID:
                    return True
            except OSError:
                continue
    return False


def _detect_linux() -> list[DetectedSensor]:
    """Linux: sysfs USB + V4L2 scan."""
    sensors: list[DetectedSensor] = []

    # 1. Check /sys/bus/usb/devices/*/idVendor for Leap Motion (f182)
    for vid_file in glob.glob("/sys/bus/usb/devices/*/idVendor"):
        try:
            vid = _read_sysfs(vid_file)
        except OSError:
            continue
        if vid == LEAP_MOTION_VID:
            parent = os.path.dirname(vid_file)
            try:
                pid = _read_sysfs(os.path.join(parent, "idProduct"))
            except OSError:
                pid = ""
            product_path = os.path.join(parent, "product")
            if os.path.exists(product_path):
                try:
                    name = _read_sysfs(product_path)
                except OSError:
                    name = "Leap Motion"
            else:
                name = "Leap Motion"
            sensors.append(
                DetectedSensor(
                    name=name,
                    type="leap_motion",
                    device_path=parent,
                    usb_vid=vid,
                    usb_pid=pid,
                )
            )

    # 2. Check /sys/class/video4linux/video*/name for webcams
    #    Skip V4L2 nodes that belong to the Leap (already listed above).
    for dev in sorted(glob.glob("/sys/class/video4linux/video*")):
        devname = os.path.basename(dev)
        devpath = f"/dev/{devname}"

        name_file = os.path.join(dev, "name")
        if os.path.exists(name_file):
            try:
                cam_name = _read_sysfs(name_file)
            except OSError:
                cam_name = devname
        else:
            cam_name = devname

        if _is_leap_v4l2(dev):
            continue

        sensors.append(
            DetectedSensor(name=cam_name, type="webcam", device_path=devpath)
        )

    return sensors


# ---------------------------------------------------------------------------
# Windows helpers
# ---------------------------------------------------------------------------

def _detect_windows() -> list[DetectedSensor]:
    """Windows: WMI USB device scan + DirectShow webcam listing.

    Uses PowerShell/WMI for USB (no pip dependencies), and optionally
    OpenCV's DirectShow backend for webcam enumeration.
    """
    sensors: list[DetectedSensor] = []

    # 1. Scan USB devices via PowerShell + WMI for the Leap Motion Controller
    try:
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                (
                    "Get-WmiObject Win32_PnPEntity | "
                    "Where-Object { $_.DeviceID -match 'USB' -and $_.DeviceID -match 'VID_F182' } | "
                    "Select-Object Name, DeviceID | "
                    "ForEach-Object { $_.Name + '|' + $_.DeviceID }"
                ),
            ],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            name = parts[0].strip() if parts else "Leap Motion"
            device_id = parts[1].strip() if len(parts) > 1 else ""
            # Extract VID/PID from DeviceID like USB\VID_F182&PID_0003\...
            vid, pid = "", ""
            for tok in device_id.replace("\\", "&").split("&"):
                tok_up = tok.upper()
                if tok_up.startswith("VID_"):
                    vid = tok_up[4:].lower()
                elif tok_up.startswith("PID_"):
                    pid = tok_up[4:].lower()
            sensors.append(
                DetectedSensor(
                    name=name,
                    type="leap_motion",
                    device_path=device_id,
                    usb_vid=vid,
                    usb_pid=pid,
                )
            )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # 2. Webcam scan: try OpenCV's DirectShow enumeration (best-effort)
    try:
        import cv2
        for idx in range(10):
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                # cv2 doesn't expose device names via DirectShow easily;
                # use a generic name with the index.
                sensors.append(
                    DetectedSensor(
                        name=f"Camera {idx}",
                        type="webcam",
                        device_path=f"dshow:{idx}",
                    )
                )
                cap.release()
            else:
                break  # no more cameras
    except ImportError:
        pass

    return sensors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_sensors() -> list[DetectedSensor]:
    """Scan USB bus and video devices for sensors OpenMotion can use.

    Cross-platform: Linux (sysfs + V4L2), Windows (WMI + DirectShow).
    Returns an empty list on unsupported platforms.
    """
    if sys.platform == "linux":
        return _detect_linux()
    elif sys.platform == "win32":
        return _detect_windows()
    else:
        return []


def print_sensors() -> None:
    found = detect_sensors()
    if not found:
        print("  (no sensors detected)")
        return
    for s in found:
        suffix = f" (USB {s.usb_vid}:{s.usb_pid})" if s.usb_vid else ""
        print(f"  [{s.type}] {s.name} at {s.device_path}{suffix}")


if __name__ == "__main__":
    print_sensors()
