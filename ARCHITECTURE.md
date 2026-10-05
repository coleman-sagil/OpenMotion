# OpenMotion — Stack Decision: which hand-tracking engine, and why

This records *why* OpenMotion tracks hands with Monado's **Mercury** engine
rather than maturing our own pipeline or staying on Ultraleap's closed stack —
and what that means for the niri integration (which API niri consumes).

It is a decision record, not a how-to. For *how* the driver and pipeline work,
read `README.md`.

## The three engines we evaluated

The irreducible work — the part only we can do — is the **driver**: USB
bring-up, calibrated stereo capture, exposure/LED control. That is done and
validated. The driver produces 1280×240 column-interleaved stereo IR frames.
The open question was always: *what turns those frames into a hand skeleton?*

Three candidates:

| Engine | What it is | Verdict |
|---|---|---|
| **Our pipeline** (`tracking/skeleton.py`, `kinematic.py`) | Mercury's ONNX nets + our own detection/coast/triangulation/scipy kinematics | Works, validated against a ruler — but a fraction of Mercury's quality, especially under occlusion. **Demoted to reference/validation.** |
| **Ultraleap Gemini** (closed `leapd` + 650 MB neural net) | The official stack | Low jitter, but **closed, unmaintained, `repo.ultraleap.com` is dead**, and fails badly under occlusion. Kept only as a comparison baseline + for its control panel. |
| **Monado Mercury** (`ht` driver: 2 ONNX nets + `kine_lm` optimizer) | Open (CC-BY-SA 4.0 models, BSL engine), actively developed for XR hand tracking | **Chosen.** Model-based kinematics → excellent occlusion handling. Fully open, so we can upstream the driver and pull engine improvements downstream. |

### Why Mercury over our own pipeline

Our scipy kinematics was the right *learning* exercise and remains the
algorithm spec / validation oracle. But Mercury is a mature, model-based
optimizer (Levenberg–Marquardt over a real hand model with joint limits). On
the live device it was, in the user's words, *"100× better than what we made"*
— the difference is most visible exactly where it matters for gestures:
**occlusion** (a fist, a pinch, a hand turned edge-on). Maturing our own to
that level would be re-deriving years of Mercury's work. Reusing Mercury (as a
first-class Monado driver, *not* a fork) means we push the `leap_open` driver
upstream and rebase to inherit every Mercury improvement for free.

### Why not stay on Ultraleap

It was genuinely good to work with and is still the low-jitter champion — but
the trade that decided it:

- **Gemini**: low jitter, **bad occlusion** (consumer-tuned smoothing + an
  open-hand assumption that collapses when fingers hide each other).
- **Mercury**: more jitter, **great occlusion** (it *infers* hidden joints from
  the model).

For gesture extraction, **jitter is the cheap half** — hysteresis and
thresholding eat it. **Occlusion is the hard half** — grab and pinch *are*
self-occlusion. So Mercury wins on the axis that's expensive to fix and loses
on the one that's cheap to fix. Add that Gemini is closed and abandoned, and
the choice makes itself.

## What Mercury actually outputs

Mercury emits one struct per hand — **`xrt_hand_joint_set`** — fetched via:

```c
xrt_device_get_hand_tracking(dev, XRT_INPUT_HT_UNOBSTRUCTED_LEFT,  ts, &joints, &out_ts);
xrt_device_get_hand_tracking(dev, XRT_INPUT_HT_UNOBSTRUCTED_RIGHT, ts, &joints, &out_ts);
```

```c
struct xrt_hand_joint_set {
    struct xrt_hand_joint_value values[26];    // the 26 OpenXR joints
    enum   xrt_hand_joint_set_flags is_active; // is this hand valid this frame
    struct xrt_space_relation hand_pose;       // base pose the joints hang off
};
struct xrt_hand_joint_value {
    struct xrt_space_relation relation; // pose: vec3 position + quat orientation
                                        //       + velocities + valid flags
    float radius;                       // joint thickness
};
```

Per hand: **26 OpenXR joints** (palm, wrist, then each finger
metacarpal → proximal → [intermediate] → distal → tip), each a **position
(meters) + orientation quaternion + valid flag**. That is *all* Mercury gives.
No pinch number, no grab number, no reliable handedness — those are
*Ultraleap* concepts, not Mercury's.

## Mercury vs. LeapD: how close is the skeleton, 1:1?

At the joint level the two skeletons are **essentially the same graph** and map
joint-for-joint. The deltas are wrapping, not substance:

| | **LeapD** (`LEAP_HAND`) | **Mercury** (`xrt_hand_joint_set`) |
|---|---|---|
| Units | millimeters | meters |
| Frame | Leap (origin = device, +Z toward user) | OpenXR (−Z forward, right-handed) |
| Joints | 5 digits × 4 bones + arm + palm | 26 OpenXR joints |
| Tips | derived from distal bone | explicit `*_TIP` joints |
| **Pinch / grab** | **precomputed** (`pinch_strength`, `grab_strength`, …) | **none — you derive it** from joint distances |
| **Handedness** | reliable (`LEAP_HAND.type`) | **slot-based, unreliable** (a known Mercury TODO) |

For what **niri** actually consumes — hand present, position, pinch state,
grab state, swipe vector, which hand — the two are **1:1 interchangeable** once
three gaps are bridged: units/frame transform (trivial), derive pinch/grab
(trivial, and we control the thresholds), and a handedness heuristic for
Mercury (the one real gap).

## The LeapC question: a historical tax, not a requirement

**LeapC is Ultraleap's API. Mercury has nothing to do with it.** niri branch A1
targets LeapC only because development *started* on the official stack, before
Mercury was in the picture. So there's a fork in the road:

- **Option A — keep LeapC (a shim).** Translate `xrt_hand_joint_set` →
  `LEAP_HAND`. niri branch A1 changes *nothing*; it keeps calling
  `LeapPollConnection()` and gets Mercury underneath. Bonus: the same shim can
  drive Ultraleap's control panel, and lets us swap Mercury ↔ LeapD beneath
  niri. Cost: we emulate a closed API's quirks (frame, pinch curves,
  handedness) on top of an open one — a translation layer that exists only for
  historical reasons.

- **Option B — drop LeapC, consume Mercury native.** Rewrite branch A1's input
  source to take `xrt_hand_joint_set` (or a thin FFI/JSON of it) directly, and
  compute pinch/grab/handedness in niri where we own them. Cleaner endpoint, no
  Ultraleap API in the loop. Cost: rip out the LeapC integration already
  written, and lose the "also drives the control panel" benefit.

**Leaning:** LeapC was the right *starting point* (the device and the official
stack both spoke it); now that Mercury is the engine, LeapC is a translation
tax. The deciding question is whether we still want the Ultraleap control panel
/ official stack as a living fallback. **If yes → shim (A). If Mercury is now
*the* one true backend → cut LeapC and feed niri the joint set directly (B).**
Not yet decided; both keep today's work alive.

## Where this leaves our own pipeline

`tracking/skeleton.py` / `kinematic.py` are **not dead** — they are the
algorithm spec and the offline validation oracle (ruler-checked depth, replay
of recorded frames). They are no longer the product. The product is:

```
Leap USB frames  ──►  leap_open Monado driver  ──►  Mercury (ht)  ──►  xrt_hand_joint_set
   (our driver)        (our driver, upstreamable)     (open engine)        │
                                                                           ▼
                                                          shim → LeapC  OR  native joints
                                                                           ▼
                                                                    niri branch A1 gestures
```

## Status (as of initial Mercury integration)

- Driver + Monado `leap_open` integration: **built, links, runs end-to-end**;
  Mercury tracks correctly after the column-deinterleave fix.
- Both Mercury and the Ultraleap stack run well on the dev machine — Mercury is
  the chosen engine; Ultraleap is retained only as a comparison baseline.

---

# OpenMotion SDK Architecture — Multi-Sensor Abstraction (Oct 2026)

OpenMotion has evolved from a single-device Leap driver into a **universal
sensor driver and detector platform**. It is the public SDK that handles
device init, frame capture, hand detection, and multi-sensor fusion.
IntuiMotion (private) consumes its output for gesture recognition, action
dispatch, and UI.

## The Four Layers

```
L0  Raw sensor bytes     (device-private — never crosses SDK boundary)
     │
     ▼
L1  Geometry: PoseFrame  ← THE SDK PUBLIC API BOUNDARY
     │
     ▼
L2  Derived semantics    (pinch/grab/extended — shared code, never per-driver)
     │
     ▼
L3  Gesture events       (IntuiMotion private — engage, swipe, pinch-click)
```

### L0 — Raw Transport (per-driver, private)

Each sensor type owns its own raw transport. This layer never leaks into
the SDK contract — it's the driver's problem.

| Sensor | L0 output | Protocol |
|--------|-----------|----------|
| Leap Motion (IR stereo) | 1280×240 8-bit gray, column-interleaved left/right | LFRM binary (our own libusb driver) |
| USB webcam (RGB mono) | 640×480 YUYV frames | V4L2 (Linux), DirectShow (Windows), AVFoundation (macOS) |
| Depth camera (future) | depth map + RGB | libfreenect2 / librealsense |
| Radar (future) | range-Doppler returns | device-specific |

### L1 — Geometry: PoseFrame (the SDK boundary)

One neutral type crosses the public boundary. Every driver emits this or
something that can be trivially converted into it.

```
PoseFrame
├─ seq: u64              monotonic frame counter
├─ t_capture_ns: u64     monotonic timestamp (nanoseconds)
├─ source_id: str        which sensor produced this
├─ caps: CapsBitmask     what this source can produce
└─ poses: [Pose]         one per tracked hand
    ├─ track_id: u32     stable across frames (NOT handedness-keyed)
    ├─ palm: Affine3     position + optional orientation
    ├─ velocity: Vec3    palm velocity (m/s)
    ├─ joints: [Joint; 26] | None
    │   └─ Joint { position: Vec3, orientation: Quat, valid: bool }
    └─ confidence: f32   0.0–1.0, source-provided
```

**Coordinate system:** right-handed, SI meters, +X right, +Y up, −Z away
from user. Matches OpenXR and ROS REP-103. Each adapter converts into
this exactly once.

**Joint ordering:** 26 entries in **OpenXR `XrHandJointEXT` order** (palm,
wrist, thumb metacarpal through tip, index through pinky). Absent joints
are `valid=false` — never interpolated, never fabricated.

**Capability flags:**
```
PALM_POSE       0x01   palm position available
PALM_ORIENT     0x02   palm orientation (not just position)
VELOCITY        0x04   velocity vector
JOINTS_FULL     0x08   all 26 joints
JOINTS_PARTIAL  0x10   some joints (e.g. 21-landmark webcam)
CHIRALITY       0x20   left/right classification
DISCRETE_EVENT  0x40   gesture-level events (e.g. radar tap)
```

### L2 — Derived Semantics (shared, never per-driver)

Computed identically for ALL sensor types in shared code. This is
essential: if each driver computed its own `pinch_strength`,
`pinch_threshold=0.85` in IntuiMotion's gesture config would silently
mean a physically different gesture on each device.

```
DerivedPose
├─ pinch_strength: f32      index-thumb distance → 0.0–1.0
├─ grab_strength: f32       finger curl aggregate → 0.0–1.0
├─ extended: [bool; 5]      per-finger extension state
├─ chirality: Left | Right
├─ chirality_confidence: f32
```

Gesture rules in IntuiMotion's YAML config gain a `requires:` key:
```yaml
pinch_click:
  requires: [JOINTS_FULL]   # ineligible on palm-only sensors
  threshold: 0.85

wave_dismiss:
  requires: [PALM_POSE, VELOCITY]  # works even on radar
  speed_min: 0.8
```

### L3 — Gesture Events (IntuiMotion private)

Engage, disengage, pinch-click, grab-fist, swipe, two-hand gestures.
This is product differentiation — stays in IntuiMotion, never in the
SDK.

## Multi-Sensor Fusion

When multiple sensors run simultaneously, fusion happens at L1 — it
merges PoseFrames from different sources into a single, higher-confidence
PoseFrame.

```
IR stereo (looking UP)     ──►  PoseFrame (source=leap)   ──┐
                                                             ├──► FusionEngine ──► merged PoseFrame
RGB webcam (looking FORWARD) ──► PoseFrame (source=cam)   ──┘
```

**Why orthogonal angles help:** the Leap sees fingers well from below
(palm-up view) but loses them when the hand rotates edge-on. The webcam
sees the hand's profile well from the front but can't resolve depth. Each
sensor's blind spot is the other's strength.

Fusion algorithm:
1. **Temporal alignment:** match frames within a configurable window
   (default 50ms) using `t_capture_ns`
2. **Hand matching:** pair hands across sources by `chirality` + spatial
   proximity of palm position
3. **Joint merge:** weighted average by source confidence × source weight;
   if only one source provides a joint, pass through with reduced
   confidence
4. **Extrinsic calibration:** a 4×4 rigid-body transform between sensor
   coordinate systems, determined once at setup time

## Source + Engine Plugin Model

A driver has two roles that can be separate:
- **Source**: produces L0 raw data (the Rust USB driver, a V4L2 capture)
- **Engine**: turns L0 into L1 joints (Mercury, MediaPipe, a custom ONNX
  model)

A sensor can register as Source-only (dumb capture, needs an external
engine), Engine-only (e.g. Mercury consuming frames from any source), or
Source+Engine (self-contained, like a webcam running its own MediaPipe
landmark detector). Each new adapter is ~80 lines of coordinate
conversion, not a re-implementation of gesture semantics.

## Cross-Platform Strategy

| Platform | USB driver | Webcam | Engine |
|----------|-----------|--------|--------|
| Linux (CachyOS/Arch) | libusb (working) | V4L2 ctypes | Mercury (ONNX) |
| Linux (Ubuntu/Pop) | libusb | V4L2 ctypes | Mercury (ONNX) |
| Windows | libusb (WinUSB) | DirectShow | Mercury (ONNX) |
| macOS | libusb (homebrew) | AVFoundation | Mercury (ONNX) |

The Rust driver cross-compiles via `rusb` (wraps libusb). The webcam
capture layer is platform-specific behind a common trait. The engine
(Mercury/MediaPipe) uses ONNX, which is inherently cross-platform.

## Licensing

OpenMotion is BSL-1.0. This is essential: IntuiMotion (private) depends
on OpenMotion, so OpenMotion's license must never force IntuiMotion open
(ruling out GPL/AGPL). BSL-1.0 is permissive enough for downstream
consumers to build proprietary products on top of the SDK.
