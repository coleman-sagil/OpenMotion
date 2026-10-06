#!/usr/bin/env python3
"""OpenMotion live hardware integration test.

Auto-detects all available sensors (Leap Motion + webcams), captures frames,
runs the ONNX hand-tracking pipeline, builds PoseFrame SDK objects, derives
gesture semantics, and demonstrates multi-source fusion.

Runnable with:
    .venv/bin/python tracking/live_test.py
"""

import fcntl
import json
import mmap
import os
import select
import struct
import sys
import time
import traceback

# Ensure imports resolve against the repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

import numpy as np

from tracking.detect import detect_sensors, DetectedSensor
from tracking.webcam_solver import WebcamPoseSolver, JOINT_NAMES as WC_JOINT_NAMES, Hand
from tracking.pose_frame import (
    PoseFrame, Pose, Joint, Caps, JOINT_NAMES as SDK_JOINT_NAMES,
    derive_semantics,
)

# ---------------------------------------------------------------------------
# V4L2 ioctl numbers (64-bit Linux)
# ---------------------------------------------------------------------------
VIDIOC_QUERYCAP  = 0x80685600
VIDIOC_S_FMT     = 0xC0D05605
VIDIOC_REQBUFS   = 0xC0145608
VIDIOC_QUERYBUF  = 0xC0585609
VIDIOC_QBUF      = 0xC058560F
VIDIOC_DQBUF     = 0xC0585611
VIDIOC_STREAMON  = 0x40045612
VIDIOC_STREAMOFF = 0x40045613

V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_PIX_FMT_YUYV = 0x56595559
V4L2_FIELD_NONE = 1

WIDTH = 640
HEIGHT = 480
NUM_BUFFERS = 2
SETTLE_FRAMES = 5

MODELS_DIR = os.path.join(os.path.dirname(_REPO_ROOT), "hand-tracking-models")
if not os.path.isdir(MODELS_DIR):
    # fallback: models next to the worktree source
    MODELS_DIR = os.path.join(_REPO_ROOT, "hand-tracking-models")

DRIVER_BIN = os.path.join(
    _REPO_ROOT, "rust-driver", "target", "release", "openleap")
BRINGUP_SEQ = os.path.join(_REPO_ROOT, "rust-driver", "bringup_seq.txt")


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------
class Results:
    """Accumulates PASS/FAIL/SKIP lines for the final status table."""

    def __init__(self):
        self._rows: list[tuple[str, str, str]] = []

    def ok(self, component: str, detail: str):
        self._rows.append(("PASS", component, detail))

    def fail(self, component: str, detail: str):
        self._rows.append(("FAIL", component, detail))

    def skip(self, component: str, detail: str):
        self._rows.append(("SKIP", component, detail))

    def print_table(self):
        print()
        print("=" * 72)
        print("  INTEGRATION TEST RESULTS")
        print("=" * 72)
        for status, comp, detail in self._rows:
            tag = {"PASS": "\033[32mPASS\033[0m",
                   "FAIL": "\033[31mFAIL\033[0m",
                   "SKIP": "\033[33mSKIP\033[0m"}.get(status, status)
            print(f"  [{tag}] {comp} -- {detail}")
        print("=" * 72)


# ---------------------------------------------------------------------------
# V4L2 helpers (proven struct layout from reference script, 64-bit Linux)
# ---------------------------------------------------------------------------
# v4l2_format (208 bytes):
#   0: type (u32), 4: [pad], 8: pix.width, 12: pix.height,
#   16: pix.pixelformat, 20: pix.field, 24: pix.bytesperline,
#   28: pix.sizeimage, 32: pix.colorspace
#
# v4l2_buffer (88 bytes):
#   0: index, 4: type, 8: bytesused, 12: flags, 16: field,
#   20-55: timestamp/timecode, 56: sequence, 60: memory,
#   64: m.offset (u32 in 8-byte union), 72: length

def v4l2_querycap(fd: int) -> dict:
    """VIDIOC_QUERYCAP -- returns driver/card/bus info or raises."""
    cap_buf = bytearray(104)
    fcntl.ioctl(fd, VIDIOC_QUERYCAP, cap_buf)
    return {
        "driver": cap_buf[0:16].split(b'\x00')[0].decode(errors='replace'),
        "card":   cap_buf[16:48].split(b'\x00')[0].decode(errors='replace'),
        "bus":    cap_buf[48:80].split(b'\x00')[0].decode(errors='replace'),
    }


def v4l2_negotiate_yuyv(fd: int, width: int, height: int) -> dict:
    """S_FMT to negotiate YUYV at the requested resolution."""
    fmt_buf = bytearray(208)
    struct.pack_into("I", fmt_buf, 0, V4L2_BUF_TYPE_VIDEO_CAPTURE)
    struct.pack_into("IIII", fmt_buf, 8, width, height,
                     V4L2_PIX_FMT_YUYV, V4L2_FIELD_NONE)
    fcntl.ioctl(fd, VIDIOC_S_FMT, fmt_buf)
    return {
        "width":        struct.unpack_from("I", fmt_buf, 8)[0],
        "height":       struct.unpack_from("I", fmt_buf, 12)[0],
        "bytesperline": struct.unpack_from("I", fmt_buf, 24)[0],
        "sizeimage":    struct.unpack_from("I", fmt_buf, 28)[0],
    }


def v4l2_capture(fd: int, width: int, height: int,
                 settle: int = SETTLE_FRAMES,
                 timeout_s: float = 3.0) -> bytes | None:
    """Capture frames, keep the last one.  Returns raw YUYV bytes or None."""
    # REQBUFS
    rb_buf = bytearray(20)
    struct.pack_into("III", rb_buf, 0, NUM_BUFFERS,
                     V4L2_BUF_TYPE_VIDEO_CAPTURE, V4L2_MEMORY_MMAP)
    fcntl.ioctl(fd, VIDIOC_REQBUFS, rb_buf)
    n_bufs = struct.unpack_from("I", rb_buf, 0)[0]

    # QUERYBUF + mmap
    buffers = []
    for i in range(n_bufs):
        qb = bytearray(88)
        struct.pack_into("I", qb, 0, i)
        struct.pack_into("I", qb, 4, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into("I", qb, 60, V4L2_MEMORY_MMAP)
        fcntl.ioctl(fd, VIDIOC_QUERYBUF, qb)
        length = struct.unpack_from("I", qb, 72)[0]
        offset = struct.unpack_from("I", qb, 64)[0]
        mm = mmap.mmap(fd, length, mmap.MAP_SHARED,
                       mmap.PROT_READ | mmap.PROT_WRITE,
                       offset=offset)
        buffers.append((mm, length))

    # QBUF all
    for i in range(n_bufs):
        qb = bytearray(88)
        struct.pack_into("I", qb, 0, i)
        struct.pack_into("I", qb, 4, V4L2_BUF_TYPE_VIDEO_CAPTURE)
        struct.pack_into("I", qb, 60, V4L2_MEMORY_MMAP)
        fcntl.ioctl(fd, VIDIOC_QBUF, qb)

    # STREAMON
    type_buf = struct.pack("i", V4L2_BUF_TYPE_VIDEO_CAPTURE)
    fcntl.ioctl(fd, VIDIOC_STREAMON, type_buf)

    frame_data = None
    try:
        for frame_num in range(settle):
            r, _, _ = select.select([fd], [], [], timeout_s)
            if not r:
                print(f"    timeout waiting for frame {frame_num}")
                break

            dq = bytearray(88)
            struct.pack_into("I", dq, 4, V4L2_BUF_TYPE_VIDEO_CAPTURE)
            struct.pack_into("I", dq, 60, V4L2_MEMORY_MMAP)
            fcntl.ioctl(fd, VIDIOC_DQBUF, dq)

            idx = struct.unpack_from("I", dq, 0)[0]
            bytesused = struct.unpack_from("I", dq, 8)[0]

            mm, length = buffers[idx]
            mm.seek(0)
            nbytes = bytesused if bytesused > 0 else length
            frame_data = mm.read(nbytes)
            print(f"    frame {frame_num}: {len(frame_data)} bytes")

            # re-queue
            rq = bytearray(88)
            struct.pack_into("I", rq, 0, idx)
            struct.pack_into("I", rq, 4, V4L2_BUF_TYPE_VIDEO_CAPTURE)
            struct.pack_into("I", rq, 60, V4L2_MEMORY_MMAP)
            fcntl.ioctl(fd, VIDIOC_QBUF, rq)
    finally:
        fcntl.ioctl(fd, VIDIOC_STREAMOFF, type_buf)
        for mm, _ in buffers:
            mm.close()

    return frame_data


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def yuyv_to_gray(data: bytes, width: int, height: int) -> np.ndarray:
    """Extract Y channel from YUYV -> (H, W) uint8."""
    raw = np.frombuffer(data, dtype=np.uint8)
    expected = width * height * 2
    if len(raw) < expected:
        padded = np.zeros(expected, dtype=np.uint8)
        padded[:len(raw)] = raw
        raw = padded
    elif len(raw) > expected:
        raw = raw[:expected]
    return raw[::2].reshape(height, width)


def save_pgm(gray: np.ndarray, path: str):
    """Save grayscale as P5 PGM."""
    h, w = gray.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode("ascii"))
        f.write(gray.tobytes())


# ---------------------------------------------------------------------------
# Conversion: WebcamPoseSolver Hand -> PoseFrame SDK Pose
# ---------------------------------------------------------------------------
def hand_to_pose(hand: Hand, source_id: str, track_id: int = 0) -> Pose:
    """Convert a webcam-solver Hand (21 MediaPipe joints) into an SDK Pose.

    MediaPipe has 21 joints; the SDK has 26 (OpenXR layout with palm and
    per-finger metacarpals).  We map the 21 into the closest SDK slots and
    leave the extra metacarpals as invalid placeholders.

    Coordinates stay in pixel space (the webcam source has no intrinsic
    calibration yet so we cannot produce metres).  derive_semantics still
    works on relative geometry.
    """
    # MediaPipe -> OpenXR SDK mapping (by index)
    #   MP 0  wrist        -> SDK 1  wrist
    #   MP 1  thumb_cmc    -> SDK 2  thumb_metacarpal
    #   MP 2  thumb_mcp    -> SDK 3  thumb_proximal
    #   MP 3  thumb_ip     -> SDK 4  thumb_distal
    #   MP 4  thumb_tip    -> SDK 5  thumb_tip
    #   MP 5  index_mcp    -> SDK 7  index_proximal
    #   MP 6  index_pip    -> SDK 8  index_intermediate
    #   MP 7  index_dip    -> SDK 9  index_distal
    #   MP 8  index_tip    -> SDK 10 index_tip
    #   (middle, ring, little follow the same +5 pattern)
    mp_to_sdk = {
        0: 1,   # wrist
        1: 2, 2: 3, 3: 4, 4: 5,        # thumb
        5: 7, 6: 8, 7: 9, 8: 10,       # index
        9: 12, 10: 13, 11: 14, 12: 15,  # middle
        13: 17, 14: 18, 15: 19, 16: 20, # ring
        17: 22, 18: 23, 19: 24, 20: 25, # little
    }

    joints = [Joint() for _ in range(26)]  # all invalid by default
    for mp_idx, sdk_idx in mp_to_sdk.items():
        if mp_idx < len(hand.landmarks):
            x, y, z = float(hand.landmarks[mp_idx][0]), \
                       float(hand.landmarks[mp_idx][1]), \
                       float(hand.landmarks[mp_idx][2])
            joints[sdk_idx] = Joint(position=(x, y, z), valid=True)

    # Synthesise palm as midpoint of wrist and middle-finger MCP
    if joints[1].valid and joints[12].valid:
        wx, wy, wz = joints[1].position
        mx, my, mz = joints[12].position
        joints[0] = Joint(
            position=((wx + mx) / 2, (wy + my) / 2, (wz + mz) / 2),
            valid=True)

    # Palm position for the Pose envelope
    palm_pos = joints[0].position if joints[0].valid else (0.0, 0.0, 0.0)

    return Pose(
        track_id=track_id,
        palm_position=palm_pos,
        joints=tuple(joints),
        confidence=hand.confidence,
    )


# ---------------------------------------------------------------------------
# Per-sensor test routines
# ---------------------------------------------------------------------------
def test_webcam(sensor: DetectedSensor, results: Results,
                solver: WebcamPoseSolver | None,
                rec_dir: str) -> list[Hand]:
    """Capture from a webcam, run ONNX pipeline, save results."""
    devpath = sensor.device_path
    devname = os.path.basename(devpath)
    print(f"\n--- Webcam: {sensor.name} ({devpath}) ---")

    # Open device
    try:
        fd = os.open(devpath, os.O_RDWR | os.O_NONBLOCK)
    except OSError as exc:
        results.fail(f"Webcam open ({devname})",
                     f"cannot open {devpath}: {exc}")
        return []

    try:
        # QUERYCAP
        try:
            cap = v4l2_querycap(fd)
            print(f"  driver={cap['driver']}  card={cap['card']}")
        except OSError as exc:
            results.fail(f"Webcam QUERYCAP ({devname})", str(exc))
            return []

        # Negotiate format
        try:
            fmt = v4l2_negotiate_yuyv(fd, WIDTH, HEIGHT)
            actual_w, actual_h = fmt["width"], fmt["height"]
            print(f"  negotiated {actual_w}x{actual_h} YUYV "
                  f"(bpl={fmt['bytesperline']}, size={fmt['sizeimage']})")
        except OSError as exc:
            results.fail(f"Webcam format ({devname})",
                         f"YUYV {WIDTH}x{HEIGHT} rejected: {exc}")
            return []

        # Capture
        try:
            frame_data = v4l2_capture(fd, actual_w, actual_h)
        except OSError as exc:
            results.fail(f"Webcam capture ({devname})", str(exc))
            return []

        if frame_data is None:
            results.fail(f"Webcam capture ({devname})",
                         "no frames received (all timed out)")
            return []

        results.ok(f"Webcam capture ({devname})",
                   f"{actual_w}x{actual_h} YUYV, {len(frame_data)} bytes")

    finally:
        os.close(fd)

    # Extract grayscale
    gray = yuyv_to_gray(frame_data, actual_w, actual_h)
    print(f"  grayscale: {gray.shape} min={gray.min()} max={gray.max()} "
          f"mean={gray.mean():.1f}")

    # Save PGM
    pgm_path = os.path.join(rec_dir, f"webcam_{devname}.pgm")
    save_pgm(gray, pgm_path)
    print(f"  saved {pgm_path}")

    # Run ONNX pipeline
    if solver is None:
        results.skip(f"ONNX pipeline ({devname})", "models not available")
        return []

    try:
        t0 = time.monotonic()
        hands = solver(gray.tobytes(), actual_w, actual_h)
        dt_ms = (time.monotonic() - t0) * 1000
    except Exception as exc:
        results.fail(f"ONNX pipeline ({devname})", str(exc))
        traceback.print_exc()
        return []

    if hands:
        results.ok(f"ONNX pipeline ({devname})",
                   f"{len(hands)} hand(s) detected in {dt_ms:.0f}ms")
    else:
        results.ok(f"ONNX pipeline ({devname})",
                   f"0 hands (pipeline OK, {dt_ms:.0f}ms)")

    # PoseFrame SDK conversion + derive_semantics
    for i, hand in enumerate(hands):
        lr = "right" if hand.handedness > 0.5 else "left"
        pose = hand_to_pose(hand, source_id=devname, track_id=i)
        sem = derive_semantics(pose)
        print(f"  hand {i} ({lr}): pinch={sem['pinch_strength']:.2f}  "
              f"grab={sem['grab_strength']:.2f}  "
              f"extended={sem['extended']}")
        results.ok(f"PoseFrame SDK ({devname} hand {i})",
                   f"derive_semantics: pinch={sem['pinch_strength']:.2f}, "
                   f"grab={sem['grab_strength']:.2f}")

    # Save results JSON
    json_path = os.path.join(rec_dir, f"webcam_{devname}.json")
    json_data = []
    for i, hand in enumerate(hands):
        json_data.append({
            "hand_index": i,
            "handedness": float(hand.handedness),
            "confidence": float(hand.confidence),
            "bbox": [float(x) for x in hand.bbox],
            "landmarks": [[float(c) for c in hand.landmarks[j]]
                          for j in range(len(hand.landmarks))],
        })
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"  saved {json_path}")

    return hands


def test_leap(sensors: list[DetectedSensor], results: Results) -> bool:
    """Probe the Leap Motion: USB presence, V4L2 caps, capture attempt."""
    leap_sensors = [s for s in sensors if s.type == "leap_motion"]
    if not leap_sensors:
        results.skip("Leap Motion USB", "not detected on USB bus")
        return False

    leap = leap_sensors[0]
    print(f"\n--- Leap Motion: {leap.name} (USB {leap.usb_vid}:{leap.usb_pid}) ---")
    results.ok("Leap Motion USB",
               f"{leap.name} (USB {leap.usb_vid}:{leap.usb_pid})")

    # Probe V4L2 nodes that might belong to the Leap
    # The Leap typically appears as /dev/video0 and /dev/video1 but may vary
    leap_v4l2_found = False
    for vidnum in range(8):
        devpath = f"/dev/video{vidnum}"
        if not os.path.exists(devpath):
            continue
        try:
            fd = os.open(devpath, os.O_RDWR | os.O_NONBLOCK)
        except OSError:
            continue
        try:
            cap = v4l2_querycap(fd)
            # Check if this node is the Leap (uvcvideo driver, card matches)
            if "leap" in cap["card"].lower() or "f182" in cap["bus"].lower():
                leap_v4l2_found = True
                driver_name = cap["driver"]
                print(f"  {devpath}: driver={driver_name}  card={cap['card']}")
                if driver_name == "uvcvideo":
                    results.ok(f"Leap V4L2 ({devpath})",
                               f"uvcvideo bound, card={cap['card']}")
                else:
                    results.ok(f"Leap V4L2 ({devpath})",
                               f"driver={driver_name}, card={cap['card']}")

                # Try a quick capture (will almost certainly timeout without
                # the Rust driver bringup sequence -- that is expected)
                print(f"  attempting quick capture on {devpath} (expect timeout)...")
                try:
                    fmt = v4l2_negotiate_yuyv(fd, 640, 240)
                    os.close(fd)
                    fd = os.open(devpath, os.O_RDWR | os.O_NONBLOCK)
                    frame_data = v4l2_capture(fd, fmt["width"], fmt["height"],
                                              settle=1, timeout_s=2.0)
                    if frame_data:
                        results.ok(f"Leap capture ({devpath})",
                                   f"got {len(frame_data)} bytes!")
                    else:
                        results.skip(f"Leap capture ({devpath})",
                                     "timeout (needs bringup -- expected)")
                except OSError as exc:
                    results.skip(f"Leap capture ({devpath})",
                                 f"V4L2 error (needs bringup): {exc}")
        except OSError:
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    if not leap_v4l2_found:
        # The Leap is on USB but no V4L2 node recognized it -- uvcvideo
        # might not be bound (the Leap needs the Rust driver to claim it)
        results.skip("Leap V4L2", "no V4L2 node bound (uvcvideo not claimed)")

    # Report bringup command
    if os.path.isfile(DRIVER_BIN) and os.path.isfile(BRINGUP_SEQ):
        cmd = f"pkexec {DRIVER_BIN} stream {BRINGUP_SEQ} 15"
        print(f"\n  Bringup command (run separately with the device plugged in):")
        print(f"    {cmd}")
        results.ok("Leap driver binary", f"found at {DRIVER_BIN}")
    elif os.path.isfile(DRIVER_BIN):
        results.ok("Leap driver binary", f"found (bringup_seq.txt missing)")
    else:
        results.skip("Leap driver binary",
                     "not built -- run: cd rust-driver && cargo build --release")

    # Leap streaming requires the bringup sequence piped from the Rust driver.
    # We cannot start it here (needs pkexec), so we skip.
    results.skip("Leap streaming",
                 "needs pkexec bringup (run run_live.sh interactively)")
    return True


# ---------------------------------------------------------------------------
# Fusion demonstration
# ---------------------------------------------------------------------------
def test_fusion(webcam_hands: dict[str, list[Hand]],
                leap_present: bool,
                results: Results):
    """Demonstrate multi-source fusion if we have hands from multiple sources
    (or at least show the path with a single source)."""
    # Collect all sources that yielded hands
    sources_with_hands = {k: v for k, v in webcam_hands.items() if v}

    if len(sources_with_hands) == 0:
        results.skip("Fusion", "no hands detected from any source")
        return

    if len(sources_with_hands) < 2 and not leap_present:
        # Only one source has hands -- show how fusion would work
        src_name = next(iter(sources_with_hands))
        hands = sources_with_hands[src_name]
        print(f"\n--- Fusion (single source: {src_name}) ---")
        print(f"  {len(hands)} hand(s) from {src_name}")
        print("  With a second source, joints would be merged by weighted average:")
        print("    merged_j = (w1 * src1_j + w2 * src2_j) / (w1 + w2)")
        print("  where weights = per-source confidence * per-joint validity")

        # Build PoseFrame from the single source to prove the types work
        poses = []
        for i, hand in enumerate(hands):
            poses.append(hand_to_pose(hand, source_id=src_name, track_id=i))
        frame = PoseFrame(
            seq=0,
            t_capture_ns=int(time.time() * 1e9),
            source_id=src_name,
            caps=Caps.JOINTS_PARTIAL | Caps.CHIRALITY,
            poses=tuple(poses),
        )
        print(f"  PoseFrame: seq={frame.seq}, {len(frame.poses)} pose(s), "
              f"caps={frame.caps!r}")
        results.ok("PoseFrame construction",
                   f"{len(frame.poses)} pose(s) from {src_name}")
        results.skip("Fusion", "needs both Leap + webcam sources")
        return

    # Multiple sources with hands -- demonstrate joint merge
    print(f"\n--- Fusion ({len(sources_with_hands)} sources) ---")
    all_poses: list[tuple[str, Pose]] = []
    for src_name, hands in sources_with_hands.items():
        for i, hand in enumerate(hands):
            pose = hand_to_pose(hand, source_id=src_name, track_id=i)
            all_poses.append((src_name, pose))
            sem = derive_semantics(pose)
            print(f"  [{src_name}] hand {i}: "
                  f"pinch={sem['pinch_strength']:.2f} "
                  f"grab={sem['grab_strength']:.2f}")

    # Weighted average of matching joints (first hand from each source)
    if len(all_poses) >= 2:
        src_a, pose_a = all_poses[0]
        src_b, pose_b = all_poses[1]
        if pose_a.joints and pose_b.joints:
            merged_joints = []
            for j_idx in range(min(len(pose_a.joints), len(pose_b.joints))):
                ja = pose_a.joints[j_idx]
                jb = pose_b.joints[j_idx]
                if ja.valid and jb.valid:
                    wa = pose_a.confidence
                    wb = pose_b.confidence
                    wt = wa + wb
                    if wt > 0:
                        mx = (wa * ja.position[0] + wb * jb.position[0]) / wt
                        my = (wa * ja.position[1] + wb * jb.position[1]) / wt
                        mz = (wa * ja.position[2] + wb * jb.position[2]) / wt
                        merged_joints.append(
                            Joint(position=(mx, my, mz), valid=True))
                    else:
                        merged_joints.append(Joint())
                elif ja.valid:
                    merged_joints.append(ja)
                elif jb.valid:
                    merged_joints.append(jb)
                else:
                    merged_joints.append(Joint())

            merged_pose = Pose(
                track_id=0,
                palm_position=merged_joints[0].position
                    if merged_joints[0].valid else (0.0, 0.0, 0.0),
                joints=tuple(merged_joints),
                confidence=(pose_a.confidence + pose_b.confidence) / 2,
            )
            sem = derive_semantics(merged_pose)
            print(f"  [merged {src_a}+{src_b}]: "
                  f"pinch={sem['pinch_strength']:.2f} "
                  f"grab={sem['grab_strength']:.2f} "
                  f"extended={sem['extended']}")
            results.ok("Fusion",
                       f"merged {src_a}+{src_b}: "
                       f"pinch={sem['pinch_strength']:.2f}, "
                       f"grab={sem['grab_strength']:.2f}")
            return

    results.skip("Fusion", "joint data insufficient for merge")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    results = Results()
    ts = time.strftime("%Y%m%d_%H%M%S")
    rec_dir = os.path.join(_REPO_ROOT, "tracking", "recordings", ts)

    print("=" * 72)
    print("  OpenMotion Live Hardware Integration Test")
    print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # ---- Step 1: Sensor detection ----
    print("\n--- Step 1: Sensor Detection ---")
    sensors = detect_sensors()
    webcams = [s for s in sensors if s.type == "webcam"]
    leaps = [s for s in sensors if s.type == "leap_motion"]

    if not sensors:
        results.skip("Sensor detection", "no sensors found")
    else:
        parts = []
        if leaps:
            parts.append(f"Leap Motion")
        if webcams:
            parts.append(f"{len(webcams)} webcam(s)")
        results.ok("Sensor detection", " + ".join(parts) + " found")
        for s in sensors:
            suffix = f" (USB {s.usb_vid}:{s.usb_pid})" if s.usb_vid else ""
            print(f"  [{s.type}] {s.name} at {s.device_path}{suffix}")

    # ---- Initialize ONNX solver ----
    solver = None
    if os.path.isdir(MODELS_DIR):
        print(f"\n--- ONNX Model Init ({MODELS_DIR}) ---")
        try:
            t0 = time.monotonic()
            solver = WebcamPoseSolver(models_dir=MODELS_DIR)
            dt = time.monotonic() - t0
            results.ok("ONNX model load", f"solver initialized in {dt:.2f}s")
        except Exception as exc:
            results.fail("ONNX model load", str(exc))
            traceback.print_exc()
    else:
        results.skip("ONNX model load",
                     f"models dir not found: {MODELS_DIR}")

    # ---- Create recordings directory ----
    os.makedirs(rec_dir, exist_ok=True)
    print(f"\n  Recordings: {rec_dir}")

    # ---- Step 2: Webcam tests ----
    webcam_hands: dict[str, list[Hand]] = {}
    for sensor in webcams:
        devname = os.path.basename(sensor.device_path)
        hands = test_webcam(sensor, results, solver, rec_dir)
        webcam_hands[devname] = hands

    if not webcams:
        results.skip("Webcam capture", "no webcams detected")

    # ---- Step 3: Leap Motion test ----
    leap_present = test_leap(sensors, results)

    # ---- Step 4: Fusion test ----
    test_fusion(webcam_hands, leap_present, results)

    # ---- Final status table ----
    results.print_table()
    return 0


if __name__ == "__main__":
    sys.exit(main())
