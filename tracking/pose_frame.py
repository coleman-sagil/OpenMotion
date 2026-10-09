#!/usr/bin/env python3
"""OpenMotion SDK public types: the canonical wire format between drivers,
tracking pipelines and applications.

Every driver/source emits PoseFrame; every consumer reads PoseFrame. Semantic
gestures (pinch, grab, finger-extended) are derived HERE, not per-driver, so
every source behaves identically.

Coordinate system: right-handed, SI metres, +X right, +Y up, -Z away from the
user (OpenXR convention).

Joint order follows the 26-joint OpenXR hand-tracking layout (XR_EXT_hand_tracking):
  palm, wrist, then per finger (thumb/index/middle/ring/little):
    metacarpal, proximal, (intermediate for non-thumb | distal for thumb),
    distal, tip
"""

from dataclasses import dataclass, field
from enum import IntFlag
from typing import Optional, Tuple
import math


class Caps(IntFlag):
    """Capability flags advertising what a source actually fills."""
    PALM_POSE = 0x01
    PALM_ORIENT = 0x02
    VELOCITY = 0x04
    JOINTS_FULL = 0x08
    JOINTS_PARTIAL = 0x10
    CHIRALITY = 0x20
    DISCRETE_EVENT = 0x40


JOINT_NAMES = (
    "palm", "wrist",
    "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip",
    "index_metacarpal", "index_proximal", "index_intermediate",
    "index_distal", "index_tip",
    "middle_metacarpal", "middle_proximal", "middle_intermediate",
    "middle_distal", "middle_tip",
    "ring_metacarpal", "ring_proximal", "ring_intermediate",
    "ring_distal", "ring_tip",
    "little_metacarpal", "little_proximal", "little_intermediate",
    "little_distal", "little_tip",
)

# Finger slices into JOINT_NAMES (start index, length)
_FINGER_SLICES = {
    "thumb":  (2, 4),   # metacarpal..tip
    "index":  (6, 5),
    "middle": (11, 5),
    "ring":   (16, 5),
    "little": (21, 5),
}
_FINGER_ORDER = ("thumb", "index", "middle", "ring", "little")

# Tip indices within JOINT_NAMES for each finger
_TIP_INDICES = {
    "thumb": 5, "index": 10, "middle": 15, "ring": 20, "little": 25,
}

# Pinch threshold: distance in metres below which pinch_strength saturates
_PINCH_CLOSE_M = 0.020   # 20 mm — fingers touching
_PINCH_OPEN_M = 0.080    # 80 mm — clearly open


@dataclass(slots=True)
class Joint:
    """A single tracked joint."""
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    valid: bool = False


@dataclass(slots=True)
class Pose:
    """One tracked hand within a frame."""
    track_id: int = 0
    palm_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    joints: Optional[Tuple[Joint, ...]] = None
    confidence: float = 1.0


@dataclass(slots=True)
class PoseFrame:
    """A timestamped snapshot from a tracking source."""
    seq: int = 0
    t_capture_ns: int = 0
    source_id: str = ""
    caps: Caps = Caps(0)
    poses: Tuple[Pose, ...] = ()


# ---------------------------------------------------------------------------
# Semantic derivation — the ONE canonical place for gesture heuristics
# ---------------------------------------------------------------------------

def _dist(a: Tuple[float, float, float],
          b: Tuple[float, float, float]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt((a[0] - b[0]) ** 2
                     + (a[1] - b[1]) ** 2
                     + (a[2] - b[2]) ** 2)


def _dot(a: Tuple[float, float, float],
         b: Tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a: Tuple[float, float, float],
         b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _finger_curl(joints: Tuple[Joint, ...], finger: str) -> float:
    """How curled a finger is, 0.0 = fully extended, 1.0 = fully curled.

    Measured as 1 - cos(angle) between the metacarpal->proximal direction
    and the proximal->tip direction, normalised to [0,1]. This is the same
    geometric test Mercury uses (dot product of bone directions).
    """
    start, length = _FINGER_SLICES[finger]
    tip_idx = start + length - 1
    prox_idx = start + 1   # proximal for all fingers

    meta = joints[start]
    prox = joints[prox_idx]
    tip = joints[tip_idx]

    if not (meta.valid and prox.valid and tip.valid):
        return 0.5  # unknown: report half-curled

    d1 = _sub(prox.position, meta.position)
    d2 = _sub(tip.position, prox.position)
    n1 = math.sqrt(_dot(d1, d1))
    n2 = math.sqrt(_dot(d2, d2))
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.5

    cos_a = _dot(d1, d2) / (n1 * n2)
    cos_a = max(-1.0, min(1.0, cos_a))
    # cos_a near 1.0 = straight = extended; near -1.0 = folded back
    return max(0.0, min(1.0, (1.0 - cos_a) / 2.0))


def joints_mp21_to_oxr26(positions) -> 'Tuple[Joint, ...]':
    """Map 21 MediaPipe joints to 26 OpenXR joints.

    positions: indexable of 21 (x,y,z) — tuples, lists, or numpy rows.
    Returns tuple of 26 Joint objects in JOINT_NAMES order.
    Synthesizes 5 metacarpals as wrist↔proximal midpoints and palm as
    wrist↔middle_MCP midpoint.
    """
    p = positions

    def mid(a, b):
        return ((a[0]+b[0])/2, (a[1]+b[1])/2, (a[2]+b[2])/2)

    def t(x):
        return (float(x[0]), float(x[1]), float(x[2]))

    j = [None] * 26
    j[0] = Joint(position=mid(p[0], p[9]), valid=True)   # palm
    j[1] = Joint(position=t(p[0]), valid=True)            # wrist
    # thumb: MP CMC(1) MCP(2) IP(3) tip(4) → OXR metacarpal proximal distal tip
    for oxr, mp in ((2, 1), (3, 2), (4, 3), (5, 4)):
        j[oxr] = Joint(position=t(p[mp]), valid=True)
    # index/middle/ring/little: MP has MCP PIP DIP tip (4)
    # OXR adds metacarpal (wrist↔MCP midpoint) before them (5)
    for fi, mp0 in enumerate((5, 9, 13, 17)):
        base = 6 + fi * 5
        j[base] = Joint(position=mid(p[0], p[mp0]), valid=True)      # metacarpal
        j[base+1] = Joint(position=t(p[mp0]), valid=True)            # proximal
        j[base+2] = Joint(position=t(p[mp0+1]), valid=True)          # intermediate
        j[base+3] = Joint(position=t(p[mp0+2]), valid=True)          # distal
        j[base+4] = Joint(position=t(p[mp0+3]), valid=True)          # tip
    return tuple(j)


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
    if pose.joints is None or len(pose.joints) < 26:
        return {
            "pinch_strength": 0.0,
            "grab_strength": 0.0,
            "extended": [True, True, True, True, True],
        }

    joints = pose.joints

    # --- pinch: thumb-tip to index-tip distance ---
    thumb_tip = joints[_TIP_INDICES["thumb"]]
    index_tip = joints[_TIP_INDICES["index"]]
    if thumb_tip.valid and index_tip.valid:
        d = _dist(thumb_tip.position, index_tip.position)
        pinch = 1.0 - max(0.0, min(1.0,
            (d - _PINCH_CLOSE_M) / (_PINCH_OPEN_M - _PINCH_CLOSE_M)))
    else:
        pinch = 0.0

    # --- per-finger curl and extended state ---
    curls = []
    extended = []
    for finger in _FINGER_ORDER:
        c = _finger_curl(joints, finger)
        curls.append(c)
        # A finger is "extended" when its curl is below threshold.
        # Thumb has a looser threshold (it naturally rests less straight).
        thr = 0.45 if finger == "thumb" else 0.35
        extended.append(c < thr)

    grab = sum(curls) / len(curls)

    return {
        "pinch_strength": pinch,
        "grab_strength": grab,
        "extended": extended,
    }
