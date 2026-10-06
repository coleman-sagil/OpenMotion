# Contributing to OpenMotion

OpenMotion is a pluggable driver and hand-tracking framework for
motion-sensing devices. The architecture separates **transport** (talking to
hardware) from **perception** (turning raw data into hand skeletons) so a new
sensor can be added without touching the gesture engine, and a new detection
engine can be added without touching any driver.

---

## Getting started

### Clone and submodule setup

```bash
git clone https://github.com/julianjc84/OpenMotion.git
cd OpenMotion
git clone https://gitlab.freedesktop.org/monado/utilities/hand-tracking-models.git
```

`hand-tracking-models/` is a separate git clone (gitignored). It contains
Monado's openly-licensed (CC-BY-SA 4.0) Mercury ONNX models and the MediaPipe
palm/landmark models that `WebcamPoseSolver` uses. Never vendor it into this
repo.

### Build

```bash
# Rust driver
( cd rust-driver && cargo build && cargo clippy --all-targets )

# Python pipeline (syntax check)
.venv/bin/python -m py_compile tracking/*.py
```

If you don't have the virtualenv yet:

```bash
python -m venv --system-site-packages .venv
.venv/bin/pip install onnxruntime opencv-python-headless numpy scipy
```

### Run tests

```bash
( cd tracking && ../.venv/bin/python -m pytest test_pose_frame.py test_webcam_solver.py test_extrinsic_cal.py test_detect.py -v )
```

The tests are standard unittest-based and can also be run individually:

```bash
( cd tracking && ../.venv/bin/python -m unittest test_pose_frame -v )
```

### Live test

```bash
./run_live.sh
```

This script stops the closed `ultraleap-hand-tracking-service`, then runs the
Python tracking pipeline under polkit. The device must be plugged in. The
closed service is left stopped on exit -- do not restart it.

Offline analysis without the device: `tracking/recordings/<ts>/` directories
(PGM frames + `results.jsonl`) can be replayed through `skeleton.run()`.

---

## Architecture: the four-layer model

OpenMotion uses a strict four-layer separation. The key boundary is between
L1 (the SDK public API) and everything above or below it.

```
L0  Raw sensor bytes       (device-private -- never crosses SDK boundary)
     |
     v
L1  Geometry: PoseFrame    <-- THE SDK PUBLIC API BOUNDARY
     |
     v
L2  Derived semantics      (pinch/grab/extended -- shared code, NEVER per-driver)
     |
     v
L3  Gesture events         (IntuiMotion private -- not in this repo)
```

| Layer | Name | Owns | Lives in | Changes when... |
|-------|------|------|----------|-----------------|
| **L0** | Raw Transport | USB bulk reads, V4L2 ioctls, device bring-up, frame parsing | `rust-driver/` (Leap), `tracking/webcam_solver.py` (webcam) | A new physical interface is needed |
| **L1** | PoseFrame Geometry | The neutral pose type that crosses the public boundary | `tracking/pose_frame.py` | The joint schema or capability flags change |
| **L2** | Derived Semantics | `pinch_strength`, `grab_strength`, `extended[]` -- computed identically for ALL sensors | `derive_semantics()` in `tracking/pose_frame.py` | The shared derivation algorithm changes |
| **L3** | Gesture Events | Engage, pinch-click, grab-fist, swipe, two-hand gestures | IntuiMotion (private repo) | Gesture vocabulary changes |

---

## The L2 rule

> **`pinch_strength`, `grab_strength`, and `extended[]` MUST be computed by
> `derive_semantics()` in `tracking/pose_frame.py`. They are NEVER computed
> per-driver.**

This is the central invariant. The function signature is:

```python
# tracking/pose_frame.py

def derive_semantics(pose: Pose) -> dict:
    """Derive gesture semantics from a Pose's joint positions.

    Returns:
        {
            "pinch_strength": float 0..1 (thumb-index proximity),
            "grab_strength":  float 0..1 (average finger curl),
            "extended":       [bool, bool, bool, bool, bool]
                              (thumb, index, middle, ring, little),
        }

    This is THE canonical derivation. Drivers must not compute their own.
    """
```

**Why this matters:** if each driver computed its own `pinch_strength`,
`pinch_threshold=0.85` in a gesture config would silently mean a physically
different gesture on each device. The shared derivation ensures that 0.85 means
the same physical gesture on a Leap Motion, a webcam, or any future sensor.

The derivation works from joint positions only:
- **Pinch**: thumb-tip to index-tip Euclidean distance, linearly mapped to
  0.0--1.0 between `_PINCH_OPEN_M` (0.080 m) and `_PINCH_CLOSE_M` (0.020 m).
- **Grab**: average finger curl across all five fingers (curl is measured from
  the dot product of bone directions, matching Mercury's geometric test).
- **Extended**: per-finger, true when curl is below a threshold (0.45 for
  thumb, 0.35 for other fingers).

Drivers/sources emit raw `Pose` objects with joints. They never compute gesture
semantics. Consumers call `derive_semantics(pose)` after receiving a
`PoseFrame`.

---

## Adding a new sensor source

A source owns L0: raw transport and frame parsing. It produces data that an
engine can turn into L1 `PoseFrame` output.

### 1. Understand the existing patterns

Two sensor types exist today:

- **Leap Motion (IR stereo)**: The Rust driver (`rust-driver/`) handles USB
  bring-up and streams 1280x240 column-interleaved stereo IR frames via
  `openleap stream`. The Python pipeline (`tracking/skeleton.py`) runs
  Mercury's ONNX nets on those frames.

- **USB webcam (RGB mono)**: `tracking/webcam_solver.py` implements
  `WebcamPoseSolver`, which takes raw grayscale frames and runs MediaPipe
  ONNX models (palm detection + hand landmark) to produce detected hands.
  Sensor detection in `tracking/detect.py` scans `/sys/bus/usb/devices/` and
  `/sys/class/video4linux/` to enumerate available hardware.

### 2. Create your source module

Add a file under `tracking/` (e.g. `tracking/depth_source.py`). Your source
must:

- **Capture raw sensor data** (L0 transport).
- **Convert coordinates** into the shared frame before handing off to an
  engine. The convention is:
  - **Right-handed**, SI **metres**
  - **+X** right, **+Y** up, **-Z** away from user
  - Matches OpenXR and ROS REP-103

  Coordinate conversion belongs in the source adapter, not the engine.

- **Produce output** that can populate a `PoseFrame` with the correct joint
  layout (26 OpenXR `XrHandJointEXT` joints) and `Caps` flags.

### 3. Register with sensor detection

Update `tracking/detect.py` to recognise your hardware. The existing pattern
scans sysfs and V4L2 nodes:

```python
# tracking/detect.py

@dataclass
class DetectedSensor:
    name: str
    type: str          # "leap_motion", "webcam", "depth_camera", ...
    device_path: str
    usb_vid: str = ""
    usb_pid: str = ""
```

Add a detection clause in `detect_sensors()` that identifies your device (USB
vendor/product ID, V4L2 node attributes, etc.) and returns a `DetectedSensor`
with the appropriate `type` string.

### 4. Emit PoseFrame

Your source ultimately produces `PoseFrame` objects (from
`tracking/pose_frame.py`). Each frame carries:

```python
PoseFrame(
    seq=<monotonic counter>,
    t_capture_ns=<monotonic nanoseconds>,
    source_id="depth_camera",      # identifies your source
    caps=Caps.PALM_POSE | Caps.JOINTS_FULL | ...,
    poses=(Pose(
        track_id=<stable across frames>,
        palm_position=(x, y, z),   # metres, in the shared frame
        velocity=(vx, vy, vz),     # m/s
        joints=tuple(Joint(...) for _ in range(26)),
        confidence=0.9,
    ),),
)
```

Set `caps` honestly -- only flag capabilities your sensor actually fills.
Absent joints must have `valid=False` (never interpolated, never fabricated).

### 5. Test

Add `tracking/test_<sensor>.py`. Follow the patterns in the existing test
files (`test_pose_frame.py`, `test_detect.py`). Assert that:
- Your sensor type appears in `detect_sensors()` output (mock sysfs if needed)
- Emitted `PoseFrame` objects have the right joint count, valid caps, and
  coordinates in the shared frame

---

## Adding a new detection engine

An engine turns L0 raw frames into L1 joint positions. It is separate from the
source that captures the raw data.

### Engine pattern: WebcamPoseSolver

The reference engine is `WebcamPoseSolver` in `tracking/webcam_solver.py`. It
is a callable class:

```python
# tracking/webcam_solver.py

class WebcamPoseSolver:
    """Detect hands in grayscale frames via MediaPipe ONNX models."""

    def __init__(self, models_dir: Optional[str] = None):
        # Loads palm_detection_MEDIAPIPE.onnx and hand_landmark_MEDIAPIPE.onnx
        # from hand-tracking-models/ (deferred import of onnxruntime)
        ...

    def __call__(self, gray_frame_bytes: bytes, width: int,
                 height: int) -> List[Hand]:
        # 1. Palm detection (SSD with 896 anchors)
        # 2. NMS
        # 3. Per-detection: crop, landmark regression (21 joints)
        # 4. Returns Hand objects with landmarks in frame-pixel coords
        ...
```

The pipeline is: grayscale frame -> palm detection (BlazePalm SSD, 128x128) ->
NMS -> per-palm crop -> landmark regression (21 joints, 224x224) -> `Hand`
objects with frame-pixel coordinates and handedness.

### Writing a new engine

1. **Create `tracking/<engine>.py`**. Your engine should be a callable that
   takes raw sensor data and returns detected poses.

2. **ONNX model placement**: put models in `hand-tracking-models/`. This
   directory is a separate git clone (CC-BY-SA 4.0 for Monado models). Your
   models must have a compatible license.

3. **Defer heavy imports**: `onnxruntime` and other inference dependencies are
   imported inside `__init__`, not at module scope. This ensures that
   `import tracking.your_engine` does not fail on platforms where the
   inference runtime is unavailable.

4. **Engine output -> PoseFrame**: the engine's raw output (e.g. pixel-space
   landmarks) must be converted into the shared coordinate frame (metres,
   right-handed, +X right, +Y up, -Z away) before populating `Joint`
   objects in a `Pose`. This coordinate conversion happens in the source
   adapter that calls the engine, not in the engine itself.

5. **The engine does NOT compute semantics.** Gesture semantics
   (`pinch_strength`, `grab_strength`, `extended[]`) come from
   `derive_semantics()` in `pose_frame.py`, called after joints are
   populated. This is the L2 rule.

### For stereo-IR sources (Leap Motion)

The Leap path uses a different engine signature. The Rust driver streams raw
stereo IR frames; the Python pipeline in `tracking/skeleton.py` runs Mercury's
ONNX nets (detection + keypoint) per eye, then triangulates into 3D. A new
stereo engine would follow the same pattern: take left/right eye images plus
stereo calibration, return 3D joint positions.

---

## Code style

### No vendor code, blobs, or service dependencies

OpenMotion contains no Ultraleap SDK code, no closed binaries, no calls to
`leapd` or the `ultraleap-hand-tracking-service`. The protocol knowledge was
obtained by observing USB traffic (see `reverse-engineering/FINDINGS.md`).
This is a hard rule: do not introduce any vendor SDK dependency.

### No vendor imports at module scope

Heavy or platform-specific imports (`onnxruntime`, `cv2`, device-specific
libraries) are deferred to `__init__` or behind lazy-load gates. Top-level
modules must be importable on any platform with just the standard library.

### Coordinate conversion in the source, not the engine

The source knows its sensor's native coordinate frame. The engine works in the
shared coordinate system (or in pixel space, with the source converting
afterward). Do not leak sensor-specific axes into engine code.

### No lint suppressions

No `#[allow(...)]` in Rust, no `# noqa` to silence real warnings. Fix lints
honestly.

### License: BSL-1.0

All contributions must be compatible with the Boost Software License 1.0. This
is essential: BSL-1.0 matches Monado (so the `leap_open` driver stays
upstreamable), is GPL-compatible (so the same code can feed niri), and is
permissive enough for downstream consumers to build proprietary products on
top of the SDK. See `LICENSING.md` for the full posture.
