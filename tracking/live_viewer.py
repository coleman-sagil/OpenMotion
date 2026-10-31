#!/usr/bin/env python3
"""OpenMotion Viewer — live hand tracking from any camera.

Uses V4L2 raw capture (Linux) or OpenCV (cross-platform) → WebcamPoseSolver
(ONNX BlazePalm + landmarks) → PoseFrame SDK types (26-joint OpenXR layout +
derive_semantics) → GTK4/Adw display.

Run:
  .venv/bin/python tracking/live_viewer.py
  .venv/bin/python tracking/live_viewer.py --device /dev/video2
  python -m tracking --viewer
"""

import json
import os
import struct
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pose_frame import joints_mp21_to_oxr26, Pose, derive_semantics  # noqa: E402
from device_registry import (  # noqa: E402
    DeviceManager, LEAP_MOTION, WEBCAM, discover_devices,
)
from webcam_solver import WebcamPoseSolver  # noqa: E402

import gi  # noqa: E402
gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
try:
    gi.require_version("Adw", "1")
    from gi.repository import Adw
    HAS_ADW = True
except (ValueError, ImportError):
    HAS_ADW = False
from gi.repository import GLib, Gdk, Gtk, Gio  # noqa: E402

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
    """Captures YUYV frames from a V4L2 device, extracts grayscale. Linux-only."""

    def __init__(self, devpath, width=640, height=480):
        super().__init__(daemon=True)
        self.devpath = devpath
        self.width, self.height = width, height
        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.total = 0

    def run(self):
        import fcntl
        import mmap as mmap_mod
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


class CvCapture(threading.Thread):
    """OpenCV capture fallback — works on Linux, Windows, macOS."""

    def __init__(self, device_index=0, width=640, height=480):
        super().__init__(daemon=True)
        self.device_index = device_index
        self.width, self.height = width, height
        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.total = 0

    def run(self):
        cap = cv2.VideoCapture(self.device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        counter = 0
        try:
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    time.sleep(0.01)
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                counter += 1
                with self.lock:
                    self.latest = (counter, gray)
                    self.total += 1
        finally:
            cap.release()

    def stats(self):
        with self.lock:
            return self.total

    def get(self):
        with self.lock:
            return self.latest

    def stop(self):
        self.running = False


class WebcamInference(threading.Thread):
    """Runs WebcamPoseSolver on grayscale frames, produces annotated RGB."""

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


def _make_reader(device):
    """Create the right capture thread for a device."""
    if sys.platform == "linux" and device.device_path.startswith("/dev/video"):
        return V4L2Reader(device.device_path)
    idx = 0
    path = device.device_path
    if path.startswith("/dev/video"):
        try:
            idx = int(path.replace("/dev/video", ""))
        except ValueError:
            pass
    return CvCapture(device_index=idx)


AppBase = Adw.Application if HAS_ADW else Gtk.Application
WinBase = Adw.ApplicationWindow if HAS_ADW else Gtk.ApplicationWindow


class Viewer(AppBase):
    def __init__(self):
        super().__init__(application_id="org.openmotion.Viewer")
        self._devmgr = DeviceManager()
        self._webcam_dev = None
        self.reader = None
        self.infer = None
        self._fps_t = self._dev_t = 0
        self._fps_n = self._dev_prev = 0
        self._fps = self._dev_fps = 0.0
        self._last_shown_n = -1
        self._gain = 1.0
        self._recording = False
        self._rec_dir = None
        self._rec_seq = 0

    def do_activate(self):
        win = WinBase(application=self, title="OpenMotion Viewer")
        win.set_default_size(1100, 680)

        # -- outer layout --
        outer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)

        # == LEFT: device sidebar ==
        sidebar = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        sidebar.set_size_request(220, -1)
        sidebar.add_css_class("background")
        sidebar.set_margin_start(6)
        sidebar.set_margin_top(6)
        sidebar.set_margin_bottom(6)

        side_label = Gtk.Label(label="Devices")
        side_label.add_css_class("title-4")
        side_label.set_xalign(0)
        side_label.set_margin_start(6)
        sidebar.append(side_label)

        self._device_list = Gtk.ListBox()
        self._device_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self._device_list.add_css_class("boxed-list")
        self._device_list.connect("row-selected", self._on_device_selected)
        scroll_dev = Gtk.ScrolledWindow()
        scroll_dev.set_vexpand(True)
        scroll_dev.set_child(self._device_list)
        sidebar.append(scroll_dev)

        rescan_btn = Gtk.Button(label="Rescan Devices")
        rescan_btn.connect("clicked", lambda _: self._rescan_devices())
        rescan_btn.set_margin_start(6)
        rescan_btn.set_margin_end(6)
        rescan_btn.set_margin_bottom(6)
        sidebar.append(rescan_btn)

        sep_l = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        outer.append(sidebar)
        outer.append(sep_l)

        # == CENTER: camera view + controls ==
        center = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        center.set_hexpand(True)

        # header bar
        if HAS_ADW:
            hb = Adw.HeaderBar()
            hb.set_title_widget(Gtk.Label(label="OpenMotion Viewer"))
        else:
            hb = Gtk.HeaderBar()
            hb.set_title_widget(Gtk.Label(label="OpenMotion Viewer"))

        self._fusion_toggle = Gtk.ToggleButton(label="Fusion")
        self._fusion_toggle.set_tooltip_text("Show fusion config panel")
        self._fusion_toggle.connect("toggled", self._on_fusion_toggled)
        hb.pack_end(self._fusion_toggle)
        center.append(hb)

        self.pic = Gtk.Picture()
        self.pic.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.pic.set_vexpand(True)
        self.pic.set_hexpand(True)
        center.append(self.pic)

        # status line
        self.status = Gtk.Label(label="detecting sensors…")
        self.status.set_xalign(0.0)
        self.status.set_margin_start(8)
        self.status.set_margin_end(8)
        center.append(self.status)

        # bottom bar
        bot = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        bot.set_margin_start(8)
        bot.set_margin_end(8)
        bot.set_margin_top(4)
        bot.set_margin_bottom(8)

        bot.append(Gtk.Label(label="gain"))
        gain = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 1.0, 4.0, 0.5)
        gain.set_value(1.0)
        gain.set_size_request(110, -1)
        gain.set_draw_value(True)
        gain.connect("value-changed", lambda s: setattr(self, "_gain", s.get_value()))
        bot.append(gain)

        snap = Gtk.Button(label="Snapshot")
        snap.set_tooltip_text("Save annotated frame + info JSON")
        snap.connect("clicked", self._on_snapshot)
        bot.append(snap)

        self._rec_btn = Gtk.ToggleButton(label="Record")
        self._rec_btn.set_tooltip_text("Record frames to tracking/recordings/")
        self._rec_btn.connect("toggled", self._on_record_toggled)
        bot.append(self._rec_btn)

        center.append(bot)
        outer.append(center)

        # == RIGHT: fusion sidebar (collapsible) ==
        sep_r = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self._sep_r = sep_r
        sep_r.set_visible(False)
        outer.append(sep_r)

        self._fusion_panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._fusion_panel.set_size_request(220, -1)
        self._fusion_panel.set_visible(False)
        self._fusion_panel.set_margin_end(6)
        self._fusion_panel.set_margin_top(6)
        self._fusion_panel.set_margin_bottom(6)

        fp_label = Gtk.Label(label="Fusion Config")
        fp_label.add_css_class("title-4")
        fp_label.set_xalign(0)
        fp_label.set_margin_start(6)
        self._fusion_panel.append(fp_label)

        self._fusion_panel.append(Gtk.Label(label="Topology"))
        self._topo_dd = Gtk.DropDown.new_from_strings(
            ["single", "dual", "quad", "room"])
        self._topo_dd.set_margin_start(6)
        self._topo_dd.set_margin_end(6)
        self._fusion_panel.append(self._topo_dd)

        self._fusion_devices_label = Gtk.Label(label="No devices in fusion")
        self._fusion_devices_label.set_xalign(0)
        self._fusion_devices_label.set_margin_start(6)
        self._fusion_devices_label.set_margin_top(12)
        self._fusion_devices_label.set_wrap(True)
        self._fusion_panel.append(self._fusion_devices_label)

        outer.append(self._fusion_panel)

        win.set_content(outer)
        win.connect("close-request", self._on_close)
        win.present()

        self._fps_t = self._dev_t = GLib.get_monotonic_time()
        self._rescan_devices()
        GLib.timeout_add(20, self._tick)

    def _rescan_devices(self):
        self._stop_capture()
        devices = self._devmgr.scan()

        while True:
            row = self._device_list.get_row_at_index(0)
            if row is None:
                break
            self._device_list.remove(row)

        if not devices:
            row = Gtk.ListBoxRow()
            row.set_child(Gtk.Label(label="No sensors found"))
            row.set_activatable(False)
            self._device_list.append(row)
            self.status.set_text("No sensors found — plug in and hit Rescan")
            return

        preselect_idx = 0
        for i, dev in enumerate(devices):
            tag = f" ({dev.vid}:{dev.pid})" if dev.vid else ""
            label = Gtk.Label(label=f"{dev.name}{tag}")
            label.set_xalign(0)
            label.set_margin_start(6)
            label.set_margin_top(4)
            label.set_margin_bottom(4)
            row = Gtk.ListBoxRow()
            row.set_child(label)
            self._device_list.append(row)
            if self._webcam_dev and dev.device_path == self._webcam_dev:
                preselect_idx = i

        self._device_list.select_row(
            self._device_list.get_row_at_index(preselect_idx))

        n_devs = len(devices)
        self._fusion_devices_label.set_text(
            f"{n_devs} device{'s' if n_devs != 1 else ''} available for fusion")

    def _on_device_selected(self, listbox, row):
        if row is None:
            return
        idx = row.get_index()
        devices = self._devmgr.devices
        if idx >= len(devices):
            return
        self._stop_capture()
        dev = devices[idx]
        self._start_source(dev)

    def _on_fusion_toggled(self, btn):
        visible = btn.get_active()
        self._fusion_panel.set_visible(visible)
        self._sep_r.set_visible(visible)

    def _stop_capture(self):
        if self.infer:
            self.infer.stop()
            self.infer = None
        if self.reader:
            self.reader.stop()
            self.reader = None
        self._last_shown_n = -1

    def _start_source(self, device):
        self.status.set_text(f"Opening {device.name} ({device.device_path})…")
        try:
            self.reader = _make_reader(device)
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
                self._fps_t = now
                self._fps_n = 0

            if self._recording and self._rec_dir:
                self._save_rec_frame(rgb, info)

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
            rec = " [REC]" if self._recording else ""
            self.status.set_text(
                f"cam {self._dev_fps:.0f} fps | infer {self._fps:.0f} fps | "
                f"hands={info.get('hands', 0)} {lr} conf={kc:.2f}{sem_str}{rec}")
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

    def _on_record_toggled(self, btn):
        if btn.get_active():
            ts = time.strftime("%Y%m%d-%H%M%S")
            self._rec_dir = os.path.join(HERE, "recordings", ts)
            os.makedirs(self._rec_dir, exist_ok=True)
            self._rec_seq = 0
            self._recording = True
        else:
            self._recording = False
            if self._rec_dir:
                self.status.set_text(
                    f"recording stopped → recordings/{os.path.basename(self._rec_dir)}/")
            self._rec_dir = None

    def _save_rec_frame(self, rgb, info):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        fname = os.path.join(self._rec_dir, f"{self._rec_seq:06d}.pgm")
        with open(fname, "wb") as f:
            f.write(f"P5\n{w} {h}\n255\n".encode())
            f.write(gray.tobytes())
        results_path = os.path.join(self._rec_dir, "results.jsonl")
        with open(results_path, "a") as f:
            f.write(json.dumps({"seq": self._rec_seq, **info}) + "\n")
        self._rec_seq += 1

    def _on_close(self, *_):
        self._recording = False
        if self.infer:
            self.infer.stop()
        if self.reader:
            self.reader.stop()
        return False


def main():
    import argparse
    p = argparse.ArgumentParser(description="OpenMotion live hand-tracking viewer")
    p.add_argument("--device",
                   help="Device path to open first (auto-detected if omitted)")
    args = p.parse_args()
    app = Viewer()
    if args.device:
        app._webcam_dev = args.device
    app.run(None)


if __name__ == "__main__":
    main()
