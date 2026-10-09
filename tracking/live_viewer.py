#!/usr/bin/env python3
"""OpenMotion Viewer — live hand tracking from any V4L2 webcam.

Uses V4L2 raw capture → WebcamPoseSolver (ONNX BlazePalm + landmarks) →
PoseFrame SDK types (26-joint OpenXR layout + derive_semantics) → GTK4 display.

Run:
  .venv/bin/python tracking/live_viewer.py
  .venv/bin/python tracking/live_viewer.py --device /dev/video2
"""

import fcntl
import json
import mmap as mmap_mod
import os
import struct
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_frame import joints_mp21_to_oxr26, Pose, derive_semantics  # noqa: E402
from detect import detect_sensors  # noqa: E402
from webcam_solver import WebcamPoseSolver  # noqa: E402

import gi  # noqa: E402
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import GLib, Gdk, Gtk  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# ---- V4L2 ioctl constants (verified 64-bit Linux offsets) ----
VIDIOC_S_FMT     = 0xC0D05605
VIDIOC_REQBUFS   = 0xC0145608
VIDIOC_QUERYBUF  = 0xC0585609
VIDIOC_QBUF      = 0xC058560F
VIDIOC_DQBUF     = 0xC0585611
VIDIOC_STREAMON  = 0x40045612
VIDIOC_STREAMOFF = 0x40045613
V4L2_PIX_FMT_YUYV = 0x56595559


class V4L2Reader(threading.Thread):
    """Captures YUYV frames from a V4L2 device, extracts grayscale."""

    def __init__(self, devpath, width=640, height=480):
        super().__init__(daemon=True)
        self.devpath = devpath
        self.width, self.height = width, height
        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.total = 0

    def run(self):
        import select as sel_mod
        fd = os.open(self.devpath, os.O_RDWR | os.O_NONBLOCK)
        try:
            fmt = bytearray(208)
            struct.pack_into("I", fmt, 0, 1)
            struct.pack_into("IIII", fmt, 8, self.width, self.height,
                             V4L2_PIX_FMT_YUYV, 1)
            fcntl.ioctl(fd, VIDIOC_S_FMT, fmt)
            w = struct.unpack_from("I", fmt, 8)[0]
            h = struct.unpack_from("I", fmt, 12)[0]
            rb = bytearray(20)
            struct.pack_into("III", rb, 0, 4, 1, 1)
            fcntl.ioctl(fd, VIDIOC_REQBUFS, rb)
            n_bufs = struct.unpack_from("I", rb, 0)[0]
            bufs = []
            for i in range(n_bufs):
                qb = bytearray(88)
                struct.pack_into("I", qb, 0, i)
                struct.pack_into("I", qb, 4, 1)
                struct.pack_into("I", qb, 60, 1)
                fcntl.ioctl(fd, VIDIOC_QUERYBUF, qb)
                length = struct.unpack_from("I", qb, 72)[0]
                offset = struct.unpack_from("I", qb, 64)[0]
                mm = mmap_mod.mmap(fd, length, mmap_mod.MAP_SHARED,
                                   mmap_mod.PROT_READ | mmap_mod.PROT_WRITE,
                                   offset=offset)
                bufs.append((mm, length))
            for i in range(n_bufs):
                qb = bytearray(88)
                struct.pack_into("I", qb, 0, i)
                struct.pack_into("I", qb, 4, 1)
                struct.pack_into("I", qb, 60, 1)
                fcntl.ioctl(fd, VIDIOC_QBUF, qb)
            fcntl.ioctl(fd, VIDIOC_STREAMON, struct.pack("i", 1))
            counter = 0
            while self.running:
                r, _, _ = sel_mod.select([fd], [], [], 1.0)
                if not r:
                    continue
                dq = bytearray(88)
                struct.pack_into("I", dq, 4, 1)
                struct.pack_into("I", dq, 60, 1)
                fcntl.ioctl(fd, VIDIOC_DQBUF, dq)
                idx = struct.unpack_from("I", dq, 0)[0]
                bytesused = struct.unpack_from("I", dq, 8)[0]
                mm, length = bufs[idx]
                mm.seek(0)
                raw = np.frombuffer(mm.read(bytesused or length), dtype=np.uint8)
                expected = w * h * 2
                if len(raw) >= expected:
                    gray = raw[:expected:2].reshape(h, w)
                else:
                    gray = np.zeros((h, w), np.uint8)
                counter += 1
                with self.lock:
                    self.latest = (counter, gray.copy())
                    self.total += 1
                rq = bytearray(88)
                struct.pack_into("I", rq, 0, idx)
                struct.pack_into("I", rq, 4, 1)
                struct.pack_into("I", rq, 60, 1)
                fcntl.ioctl(fd, VIDIOC_QBUF, rq)
            fcntl.ioctl(fd, VIDIOC_STREAMOFF, struct.pack("i", 1))
            for mm, _ in bufs:
                mm.close()
        except Exception as e:
            print(f"V4L2 error: {e}", file=sys.stderr)
        finally:
            os.close(fd)

    def stats(self):
        with self.lock:
            return self.total

    def get(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.running = False


class WebcamInference(threading.Thread):
    """Runs WebcamPoseSolver on V4L2 grayscale frames, produces annotated RGB."""

    def __init__(self, reader):
        super().__init__(daemon=True)
        self.reader = reader
        models_dir = os.path.join(HERE, "..", "hand-tracking-models")
        self.solver = WebcamPoseSolver(models_dir=models_dir)
        self.lock = threading.Lock()
        self.annotated = None
        self.info = {}
        self.running = True
        self.n = 0
        self._last_counter = -1

    # ponytail: single-threaded inference, per-webcam threads if throughput matters
    def run(self):
        while self.running:
            item = self.reader.get()
            if item is None or item[0] == self._last_counter:
                threading.Event().wait(0.01)
                continue
            self._last_counter, gray = item
            h, w = gray.shape
            hands = self.solver(gray.tobytes(), w, h)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            info = {"hands": len(hands), "kp_conf": 0.0}
            if hands:
                info["kp_conf"] = max(hand.confidence for hand in hands)
                for hand in hands:
                    pts = [(int(hand.landmarks[j, 0]), int(hand.landmarks[j, 1]))
                           for j in range(21)]
                    for a, b in [(0,1),(1,2),(2,3),(3,4),
                                 (0,5),(5,6),(6,7),(7,8),
                                 (0,9),(9,10),(10,11),(11,12),
                                 (0,13),(13,14),(14,15),(15,16),
                                 (0,17),(17,18),(18,19),(19,20),
                                 (5,9),(9,13),(13,17)]:
                        cv2.line(rgb, pts[a], pts[b], (0, 255, 100), 2, cv2.LINE_AA)
                    for p in pts:
                        cv2.circle(rgb, p, 3, (255, 100, 0), -1, cv2.LINE_AA)
                    # ponytail: /1000 treats pixels as pseudo-mm so pinch thresholds
                    # (~20-80mm) land in the right ballpark for 640px frames
                    oxr = joints_mp21_to_oxr26(hand.landmarks / 1000.0)
                    sem = derive_semantics(Pose(joints=oxr))
                    info["sem"] = sem
                    info["hand_lr"] = "R" if hand.handedness > 0.5 else "L"
            with self.lock:
                self.annotated = np.ascontiguousarray(rgb)
                self.info = info
                self.n += 1

    def get(self):
        with self.lock:
            if self.annotated is None:
                return None, None, 0
            return self.annotated, dict(self.info), self.n

    def stop(self):
        self.running = False


class Viewer(Gtk.Application):
    def __init__(self):
        super().__init__(application_id="org.openmotion.Viewer")
        self._webcam_dev = None
        self.reader = None
        self.infer = None
        self._fps_t = self._dev_t = 0
        self._fps_n = self._dev_prev = 0
        self._fps = self._dev_fps = 0.0
        self._last_shown_n = -1
        self._gain = 1.0
        self._sensors = []
        self._sensor_dd = None

    def do_activate(self):
        win = Gtk.ApplicationWindow(application=self, title="OpenMotion Viewer")
        win.set_default_size(900, 560)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        win.set_child(box)

        self.pic = Gtk.Picture()
        self.pic.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.pic.set_vexpand(True)
        self.pic.set_hexpand(True)
        box.append(self.pic)

        self.status = Gtk.Label(label="detecting sensors…")
        self.status.set_xalign(0.0)
        box.append(self.status)

        ctrl = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        ctrl.set_margin_start(8); ctrl.set_margin_end(8); ctrl.set_margin_bottom(8)
        box.append(ctrl)

        ctrl.append(Gtk.Label(label="sensor"))
        self._sensor_dd = Gtk.DropDown.new_from_strings(["(scanning…)"])
        self._sensor_dd.set_size_request(200, -1)
        self._sensor_dd.connect("notify::selected", self._on_sensor_changed)
        ctrl.append(self._sensor_dd)

        rescan = Gtk.Button(label="rescan")
        rescan.set_tooltip_text("Re-detect sensors (plug/unplug without restarting)")
        rescan.connect("clicked", lambda _: self._rescan_sensors())
        ctrl.append(rescan)

        ctrl.append(Gtk.Label(label="gain"))
        gain = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1.0, 4.0, 0.5)
        gain.set_value(1.0); gain.set_size_request(110, -1)
        gain.set_draw_value(True)
        gain.connect("value-changed", lambda s: setattr(self, "_gain", s.get_value()))
        ctrl.append(gain)

        snap = Gtk.Button(label="snapshot")
        snap.set_tooltip_text("Save annotated frame + info JSON to tracking/snapshots/")
        snap.connect("clicked", self._on_snapshot)
        ctrl.append(snap)

        win.connect("close-request", self._on_close)
        win.present()

        self._fps_t = self._dev_t = GLib.get_monotonic_time()
        self._rescan_sensors()
        GLib.timeout_add(20, self._tick)

    def _rescan_sensors(self):
        self._stop_capture()
        self._sensors = [s for s in detect_sensors() if s.device_path and
                         s.device_path.startswith("/dev/")]
        if not self._sensors:
            names = ["(no sensors found)"]
        else:
            names = [f"{s.name} ({s.device_path})" for s in self._sensors]
        model = Gtk.StringList.new(names)
        self._sensor_dd.set_model(model)
        preselect = 0
        if self._webcam_dev:
            for i, s in enumerate(self._sensors):
                if s.device_path == self._webcam_dev:
                    preselect = i
                    break
        self._sensor_dd.set_selected(preselect)
        if self._sensors:
            self._start_source(self._sensors[preselect].device_path)
        else:
            self.status.set_text("No sensors found — plug one in and hit rescan")

    def _on_sensor_changed(self, dd, _pspec):
        idx = dd.get_selected()
        if idx < len(self._sensors):
            self._stop_capture()
            self._start_source(self._sensors[idx].device_path)

    def _stop_capture(self):
        if self.infer:
            self.infer.stop()
            self.infer = None
        if self.reader:
            self.reader.stop()
            self.reader = None
        self._last_shown_n = -1

    def _start_source(self, devpath):
        self.status.set_text(f"Opening {devpath}…")
        try:
            self.reader = V4L2Reader(devpath)
            self.reader.start()
            self.infer = WebcamInference(self.reader)
            self.infer.start()
        except Exception as e:
            self.status.set_text(f"Sensor failed: {e}")

    def _tick(self):
        if self.infer is None:
            return True
        now = GLib.get_monotonic_time()
        if now - self._dev_t > 1_000_000 and self.reader is not None:
            total = self.reader.stats()
            self._dev_fps = (total - self._dev_prev) * 1e6 / (now - self._dev_t)
            self._dev_prev = total
            self._dev_t = now
        rgb, info, n = self.infer.get()
        if rgb is not None and n != self._last_shown_n:
            self._last_shown_n = n
            if self._gain != 1.0:
                rgb = cv2.convertScaleAbs(rgb, alpha=self._gain)
            h, w = rgb.shape[:2]
            tex = Gdk.MemoryTexture.new(
                w, h, Gdk.MemoryFormat.R8G8B8,
                GLib.Bytes.new(rgb.tobytes()), w * 3)
            self.pic.set_paintable(tex)
            self._fps_n += 1
            now = GLib.get_monotonic_time()
            if now - self._fps_t > 1_000_000:
                self._fps = self._fps_n * 1e6 / (now - self._fps_t)
                self._fps_t = now; self._fps_n = 0
            sem = info.get("sem")
            sem_str = ""
            if sem:
                ext = sem.get("extended", [])
                fingers = "[" + "".join(
                    ch if e else ch.lower()
                    for ch, e in zip("TIMRL", ext)) + "]"
                sem_str = (f" | pinch={sem['pinch_strength']:.0%}"
                           f" grab={sem['grab_strength']:.0%} {fingers}")
            lr = info.get("hand_lr", "")
            kc = info.get("kp_conf", 0.0)
            self.status.set_text(
                f"cam {self._dev_fps:.0f} fps | infer {self._fps:.0f} fps | "
                f"hands={info.get('hands', 0)} {lr} conf={kc:.2f}{sem_str}")
        elif self.reader and not self.reader.is_alive():
            self.status.set_text("webcam disconnected")
        return True

    def _on_snapshot(self, _btn):
        if self.infer is None:
            return
        rgb, info, _ = self.infer.get()
        if rgb is None:
            self.status.set_text("snapshot: no frame yet")
            return
        ts = time.strftime("%Y%m%d-%H%M%S")
        d = os.path.join(HERE, "snapshots", ts)
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "annotated.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        with open(os.path.join(d, "info.json"), "w") as f:
            json.dump(info, f, indent=1)
        self.status.set_text(f"snapshot → snapshots/{ts}/")

    def _on_close(self, *_):
        if self.infer:
            self.infer.stop()
        if self.reader:
            self.reader.stop()
        return False


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="OpenMotion live hand-tracking viewer")
    p.add_argument("--device", help="V4L2 device path to open first (auto-detected if omitted)")
    args = p.parse_args()
    app = Viewer()
    if args.device:
        app._webcam_dev = args.device
    app.run(None)
