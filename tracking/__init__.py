"""OpenMotion — open-source multi-sensor hand tracking SDK."""

from .pose_frame import (
    Caps,
    Joint,
    JOINT_NAMES,
    Pose,
    PoseFrame,
    derive_semantics,
    joints_mp21_to_oxr26,
)
from .detect import detect_sensors

try:
    from .device_registry import DeviceInfo, DeviceManager, discover_devices
except ImportError:
    pass

try:
    from .webcam_solver import WebcamPoseSolver
except ImportError:
    pass

__version__ = "0.1.0"
