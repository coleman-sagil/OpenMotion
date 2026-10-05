#!/usr/bin/env python3
"""Scan USB bus and V4L2 devices for sensors OpenMotion can use.

Linux-only: on other platforms detect_sensors() returns an empty list.
"""

import os
import glob
import sys
from dataclasses import dataclass, field


LEAP_MOTION_VID = "f182"


@dataclass
class DetectedSensor:
    name: str
    type: str          # "leap_motion", "webcam"
    device_path: str
    usb_vid: str = ""
    usb_pid: str = ""


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


def detect_sensors() -> list[DetectedSensor]:
    """Scan USB bus and V4L2 devices for sensors OpenMotion can use.

    Returns an empty list on non-Linux platforms.
    """
    if sys.platform != "linux":
        return []

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
