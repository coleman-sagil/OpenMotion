#!/usr/bin/env python3
"""Universal device discovery for OpenMotion.

Platform-aware sensor enumeration: Leap Motion, webcams, depth cameras,
and future motion-capture devices on Linux, Windows, and macOS.

Linux:   /sys/bus/usb + /sys/class/video4linux (native, zero deps)
Windows: PowerShell Get-PnpDevice fallback (no native DLL calls yet)
macOS:   stub (AVFoundation enumeration planned)
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass, field
from typing import Callable

LEAP_MOTION_VID = "f182"

LEAP_MOTION = "leap_motion"
WEBCAM = "webcam"
DEPTH_CAMERA = "depth_camera"
UNKNOWN = "unknown"


@dataclass(slots=True)
class DeviceInfo:
    """A detected sensor device."""
    id: str
    name: str
    device_type: str
    device_path: str
    vid: str = ""
    pid: str = ""
    capabilities: dict = field(default_factory=dict)


def discover_devices() -> list[DeviceInfo]:
    """Discover all connected sensors. Cross-platform."""
    if sys.platform == "linux":
        return _discover_linux()
    if sys.platform.startswith("win"):
        return _discover_windows()
    return []


def _read(path: str) -> str:
    with open(path) as f:
        return f.read().strip()


def _discover_linux() -> list[DeviceInfo]:
    devices: list[DeviceInfo] = []
    leap_usb_realpaths: set[str] = set()

    for vid_file in glob.glob("/sys/bus/usb/devices/*/idVendor"):
        try:
            vid = _read(vid_file)
        except OSError:
            continue
        if vid != LEAP_MOTION_VID:
            continue
        parent = os.path.dirname(vid_file)
        leap_usb_realpaths.add(os.path.realpath(parent))
        try:
            pid = _read(os.path.join(parent, "idProduct"))
        except OSError:
            pid = ""
        try:
            name = _read(os.path.join(parent, "product"))
        except OSError:
            name = "Leap Motion Controller"

        v4l2_nodes = []
        for vdev in sorted(glob.glob("/sys/class/video4linux/video*")):
            real = os.path.realpath(vdev)
            if any(real.startswith(lp) for lp in leap_usb_realpaths):
                v4l2_nodes.append(f"/dev/{os.path.basename(vdev)}")

        slug = os.path.basename(parent)
        devices.append(DeviceInfo(
            id=f"usb-{vid}:{pid}-{slug}",
            name=name,
            device_type=LEAP_MOTION,
            device_path=v4l2_nodes[0] if v4l2_nodes else parent,
            vid=vid, pid=pid,
            capabilities={"v4l2_nodes": v4l2_nodes, "usb_path": parent},
        ))

    for vdev in sorted(glob.glob("/sys/class/video4linux/video*")):
        real = os.path.realpath(vdev)
        if any(real.startswith(lp) for lp in leap_usb_realpaths):
            continue
        devname = os.path.basename(vdev)
        devpath = f"/dev/{devname}"
        try:
            cam_name = _read(os.path.join(vdev, "name"))
        except OSError:
            cam_name = devname

        vid = pid = ""
        parts = real.split("/")
        for i in range(len(parts)):
            vp = "/".join(parts[: i + 1]) + "/idVendor"
            if os.path.exists(vp):
                try:
                    vid = _read(vp)
                    pid = _read("/".join(parts[: i + 1]) + "/idProduct")
                except OSError:
                    pass
                break

        devices.append(DeviceInfo(
            id=devpath, name=cam_name, device_type=WEBCAM,
            device_path=devpath, vid=vid, pid=pid,
        ))

    return devices


def _discover_windows() -> list[DeviceInfo]:
    # ponytail: PowerShell probe, upgrade to SetupAPI ctypes when validated
    devices: list[DeviceInfo] = []
    try:
        import json
        import subprocess
        r = subprocess.run(
            ["powershell", "-Command",
             "Get-PnpDevice -Class Camera -Status OK "
             "| Select-Object FriendlyName,InstanceId "
             "| ConvertTo-Json"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            data = json.loads(r.stdout)
            if isinstance(data, dict):
                data = [data]
            for item in data:
                name = item.get("FriendlyName", "Camera")
                iid = item.get("InstanceId", "")
                devices.append(DeviceInfo(
                    id=iid, name=name, device_type=WEBCAM, device_path=iid,
                ))
    except Exception:
        pass
    return devices


class DeviceManager:
    """Caches device list and fires callbacks on changes."""

    def __init__(self):
        self._devices: list[DeviceInfo] = []
        self._callbacks: list[Callable[[list[DeviceInfo]], None]] = []

    def scan(self) -> list[DeviceInfo]:
        new = discover_devices()
        old_ids = {d.id for d in self._devices}
        new_ids = {d.id for d in new}
        self._devices = new
        if old_ids != new_ids:
            for cb in self._callbacks:
                try:
                    cb(new)
                except Exception:
                    pass
        return new

    def get(self, device_id: str) -> DeviceInfo | None:
        for d in self._devices:
            if d.id == device_id:
                return d
        return None

    def filter_by_type(self, device_type: str) -> list[DeviceInfo]:
        return [d for d in self._devices if d.device_type == device_type]

    def on_change(self, callback: Callable[[list[DeviceInfo]], None]):
        self._callbacks.append(callback)

    @property
    def devices(self) -> list[DeviceInfo]:
        return list(self._devices)


if __name__ == "__main__":
    print("OpenMotion Device Registry")
    print("=" * 40)
    found = discover_devices()
    if not found:
        print("  (no devices detected)")
    else:
        for d in found:
            tag = f" USB {d.vid}:{d.pid}" if d.vid else ""
            print(f"  [{d.device_type}] {d.name}{tag}")
            print(f"    path: {d.device_path}  id: {d.id}")
