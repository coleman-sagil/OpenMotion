#!/usr/bin/env python3
"""Tests for detect.py — mock sysfs trees in a tmpdir."""

import os
import sys
import unittest
from unittest import mock
from pathlib import Path

# Ensure the tracking package is importable.
sys.path.insert(0, os.path.dirname(__file__))
from detect import detect_sensors, DetectedSensor, _is_leap_v4l2


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


class TestDetectSensorsNonLinux(unittest.TestCase):
    """On non-Linux platforms, detect_sensors() must return []."""

    @mock.patch("detect.sys")
    def test_returns_empty_on_non_linux(self, mock_sys):
        mock_sys.platform = "darwin"
        self.assertEqual(detect_sensors(), [])


class TestDetectLeapMotion(unittest.TestCase):
    """Leap Motion detection via /sys/bus/usb/devices/*/idVendor."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _usb_device(self, bus_id: str, vid: str, pid: str,
                     product: str | None = None) -> str:
        """Create a fake /sys/bus/usb/devices/<bus_id>/ tree."""
        dev_dir = os.path.join(self.tmpdir, "sys", "bus", "usb",
                               "devices", bus_id)
        os.makedirs(dev_dir, exist_ok=True)
        _write(os.path.join(dev_dir, "idVendor"), vid + "\n")
        _write(os.path.join(dev_dir, "idProduct"), pid + "\n")
        if product is not None:
            _write(os.path.join(dev_dir, "product"), product + "\n")
        return dev_dir

    @mock.patch("detect.sys")
    def test_finds_leap_motion(self, mock_sys):
        mock_sys.platform = "linux"
        dev = self._usb_device("1-2", "f182", "0003",
                               product="Leap Motion Controller")
        usb_glob = os.path.join(self.tmpdir, "sys", "bus", "usb",
                                "devices", "*", "idVendor")
        v4l_glob = os.path.join(self.tmpdir, "sys", "class",
                                "video4linux", "video*")

        with mock.patch("detect.glob.glob", side_effect=lambda p: (
            sorted(Path(self.tmpdir).glob("sys/bus/usb/devices/*/idVendor"))
            if "usb" in p
            else sorted(Path(self.tmpdir).glob("sys/class/video4linux/video*"))
        )):
            sensors = detect_sensors()

        self.assertEqual(len(sensors), 1)
        s = sensors[0]
        self.assertEqual(s.type, "leap_motion")
        self.assertEqual(s.usb_vid, "f182")
        self.assertEqual(s.usb_pid, "0003")
        self.assertEqual(s.name, "Leap Motion Controller")
        self.assertEqual(s.device_path, dev)

    @mock.patch("detect.sys")
    def test_leap_without_product_file(self, mock_sys):
        mock_sys.platform = "linux"
        self._usb_device("1-3", "f182", "0003", product=None)

        with mock.patch("detect.glob.glob", side_effect=lambda p: (
            sorted(Path(self.tmpdir).glob("sys/bus/usb/devices/*/idVendor"))
            if "usb" in p
            else []
        )):
            sensors = detect_sensors()

        self.assertEqual(len(sensors), 1)
        self.assertEqual(sensors[0].name, "Leap Motion")

    @mock.patch("detect.sys")
    def test_ignores_non_leap_usb(self, mock_sys):
        mock_sys.platform = "linux"
        self._usb_device("2-1", "046d", "0825", product="Logitech Webcam")

        with mock.patch("detect.glob.glob", side_effect=lambda p: (
            sorted(Path(self.tmpdir).glob("sys/bus/usb/devices/*/idVendor"))
            if "usb" in p
            else []
        )):
            sensors = detect_sensors()

        self.assertEqual(sensors, [])


class TestDetectWebcam(unittest.TestCase):
    """Webcam detection via /sys/class/video4linux."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def _v4l_device(self, devname: str, cam_name: str,
                     parent_vid: str | None = None) -> str:
        """Create a fake /sys/class/video4linux/<devname>/ tree.

        If parent_vid is set, create a USB ancestor with that VID so the
        Leap-skip logic can be tested.
        """
        dev_dir = os.path.join(self.tmpdir, "sys", "class",
                               "video4linux", devname)

        if parent_vid is not None:
            # Build a real path the symlink resolves to, with a USB
            # ancestor containing idVendor.
            real_dir = os.path.join(self.tmpdir, "sys", "devices",
                                    "usb_parent", devname)
            os.makedirs(real_dir, exist_ok=True)
            # Put idVendor at the usb_parent level.
            _write(os.path.join(self.tmpdir, "sys", "devices",
                                "usb_parent", "idVendor"),
                   parent_vid + "\n")
            # Create the symlink.
            os.makedirs(os.path.dirname(dev_dir), exist_ok=True)
            os.symlink(real_dir, dev_dir)
        else:
            os.makedirs(dev_dir, exist_ok=True)

        _write(os.path.join(dev_dir, "name"), cam_name + "\n")
        return dev_dir

    @mock.patch("detect.sys")
    def test_finds_webcam(self, mock_sys):
        mock_sys.platform = "linux"
        self._v4l_device("video0", "USB Camera")

        with mock.patch("detect.glob.glob", side_effect=lambda p: (
            []
            if "usb" in p
            else sorted(Path(self.tmpdir).glob("sys/class/video4linux/video*"))
        )):
            sensors = detect_sensors()

        self.assertEqual(len(sensors), 1)
        s = sensors[0]
        self.assertEqual(s.type, "webcam")
        self.assertEqual(s.name, "USB Camera")
        self.assertEqual(s.device_path, "/dev/video0")

    @mock.patch("detect.sys")
    def test_skips_leap_v4l2_node(self, mock_sys):
        mock_sys.platform = "linux"
        self._v4l_device("video2", "Leap Motion", parent_vid="f182")

        with mock.patch("detect.glob.glob", side_effect=lambda p: (
            []
            if "usb" in p
            else sorted(Path(self.tmpdir).glob("sys/class/video4linux/video*"))
        )):
            sensors = detect_sensors()

        self.assertEqual(sensors, [])

    @mock.patch("detect.sys")
    def test_mixed_detection(self, mock_sys):
        """Leap USB + webcam + Leap V4L2 node: only Leap USB + webcam
        returned."""
        mock_sys.platform = "linux"

        # Leap on USB bus
        usb_dir = os.path.join(self.tmpdir, "sys", "bus", "usb",
                               "devices", "1-2")
        os.makedirs(usb_dir, exist_ok=True)
        _write(os.path.join(usb_dir, "idVendor"), "f182\n")
        _write(os.path.join(usb_dir, "idProduct"), "0003\n")
        _write(os.path.join(usb_dir, "product"),
               "Leap Motion Controller\n")

        # Webcam (no Leap parent)
        self._v4l_device("video0", "HD Pro Webcam")

        # Leap V4L2 node (should be skipped)
        self._v4l_device("video2", "Leap V4L2", parent_vid="f182")

        with mock.patch("detect.glob.glob", side_effect=lambda p: (
            sorted(Path(self.tmpdir).glob("sys/bus/usb/devices/*/idVendor"))
            if "usb" in p
            else sorted(Path(self.tmpdir).glob("sys/class/video4linux/video*"))
        )):
            sensors = detect_sensors()

        types = [s.type for s in sensors]
        self.assertIn("leap_motion", types)
        self.assertIn("webcam", types)
        # The Leap V4L2 node must not appear as a webcam.
        webcams = [s for s in sensors if s.type == "webcam"]
        self.assertEqual(len(webcams), 1)
        self.assertEqual(webcams[0].name, "HD Pro Webcam")


class TestDetectEmpty(unittest.TestCase):
    """No devices at all."""

    @mock.patch("detect.sys")
    def test_no_devices(self, mock_sys):
        mock_sys.platform = "linux"
        with mock.patch("detect.glob.glob", return_value=[]):
            sensors = detect_sensors()
        self.assertEqual(sensors, [])


if __name__ == "__main__":
    unittest.main()
