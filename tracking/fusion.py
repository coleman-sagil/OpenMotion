#!/usr/bin/env python3
"""Multi-device fusion engine for OpenMotion.

Merges hand-tracking data from multiple sensors into a single unified
PoseFrame. Supports configurable topologies — from a single device to a
room-scale array — and computes extrinsic calibration between device pairs
via the Kabsch algorithm (SVD rigid-body alignment).

Topology labels are common presets, not a class hierarchy:
  SINGLE  1 device, passthrough (no fusion)
  DUAL    2 devices facing each other (stereo coverage)
  QUAD    4 devices in a square (360° coverage)
  ROOM    arbitrary placement, full extrinsic calibration required
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from pose_frame import Caps, Joint, Pose, PoseFrame

SINGLE = "single"
DUAL = "dual"
QUAD = "quad"
ROOM = "room"


@dataclass
class FusionConfig:
    """Configuration for multi-device fusion."""
    topology: str = SINGLE
    device_ids: list[str] = field(default_factory=list)
    # (dev_a, dev_b) -> 4x4 SE3 transform from dev_b's frame into dev_a's frame
    extrinsics: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)
    merge_window_ms: float = 50.0
    # per-device confidence weight (default 1.0 for all)
    weights: dict[str, float] = field(default_factory=dict)


@dataclass
class _DeviceSlot:
    frame: PoseFrame
    timestamp: float


class FusionEngine:
    """Fuses PoseFrames from multiple devices into a single output.

    Usage:
        engine = FusionEngine(config)
        engine.add_frame("leap-1", frame1, time.monotonic())
        engine.add_frame("webcam-0", frame2, time.monotonic())
        fused = engine.get_fused_frame()
    """

    def __init__(self, config: FusionConfig):
        self.config = config
        self._slots: dict[str, _DeviceSlot] = {}
        self._ref_device: str = config.device_ids[0] if config.device_ids else ""
        self._seq = 0

    def add_frame(self, device_id: str, frame: PoseFrame, timestamp: float):
        self._slots[device_id] = _DeviceSlot(frame=frame, timestamp=timestamp)

    def get_fused_frame(self) -> Optional[PoseFrame]:
        if not self._slots:
            return None
        if self.config.topology == SINGLE or len(self._slots) == 1:
            slot = next(iter(self._slots.values()))
            return slot.frame

        now = time.monotonic()
        window_s = self.config.merge_window_ms / 1000.0
        active = {
            did: slot for did, slot in self._slots.items()
            if (now - slot.timestamp) < window_s
        }
        if not active:
            return None
        if len(active) == 1:
            return next(iter(active.values())).frame

        all_poses = self._gather_poses(active)
        if not all_poses:
            return None

        merged_pose = self._merge_poses(all_poses)
        self._seq += 1
        return PoseFrame(
            seq=self._seq,
            t_capture_ns=int(now * 1e9),
            source_id="fusion",
            caps=Caps.JOINTS_FULL,
            poses=(merged_pose,) if merged_pose else (),
        )

    def _gather_poses(self, active: dict[str, _DeviceSlot]) -> list[tuple[Pose, float]]:
        """Collect all poses with their device weights, transformed to reference frame."""
        result = []
        for did, slot in active.items():
            w = self.config.weights.get(did, 1.0)
            for pose in slot.frame.poses:
                if pose.joints and len(pose.joints) >= 26:
                    transformed = self._transform_pose(pose, did)
                    result.append((transformed, w))
        return result

    def _transform_pose(self, pose: Pose, device_id: str) -> Pose:
        """Transform a pose from device_id's coordinate frame to the reference frame."""
        if device_id == self._ref_device:
            return pose
        key = (self._ref_device, device_id)
        T = self.config.extrinsics.get(key)
        if T is None:
            return pose

        R = T[:3, :3]
        t = T[:3, 3]
        new_joints = []
        for j in pose.joints:
            if j.valid:
                p = np.array(j.position)
                p_ref = R @ p + t
                new_joints.append(Joint(
                    position=(float(p_ref[0]), float(p_ref[1]), float(p_ref[2])),
                    orientation=j.orientation,
                    valid=True,
                ))
            else:
                new_joints.append(j)
        return Pose(
            track_id=pose.track_id,
            palm_position=pose.palm_position,
            joints=tuple(new_joints),
            confidence=pose.confidence,
        )

    def _merge_poses(self, weighted_poses: list[tuple[Pose, float]]) -> Optional[Pose]:
        """Weighted average of joint positions across devices."""
        if not weighted_poses:
            return None

        n_joints = 26
        accum = np.zeros((n_joints, 3))
        weight_sum = np.zeros(n_joints)

        for pose, w in weighted_poses:
            if not pose.joints or len(pose.joints) < n_joints:
                continue
            for i, j in enumerate(pose.joints):
                if j.valid:
                    accum[i] += np.array(j.position) * w * pose.confidence
                    weight_sum[i] += w * pose.confidence

        joints = []
        for i in range(n_joints):
            if weight_sum[i] > 1e-9:
                p = accum[i] / weight_sum[i]
                joints.append(Joint(
                    position=(float(p[0]), float(p[1]), float(p[2])),
                    valid=True,
                ))
            else:
                joints.append(Joint(valid=False))

        best_conf = max(p.confidence for p, _ in weighted_poses)
        return Pose(
            joints=tuple(joints),
            confidence=best_conf,
            palm_position=joints[0].position if joints[0].valid else (0, 0, 0),
        )

    @staticmethod
    def calibrate_pair(
        observations: list[tuple[PoseFrame, PoseFrame]],
    ) -> np.ndarray:
        """Compute rigid transform from device B's frame to device A's frame.

        observations: list of (frame_from_A, frame_from_B) captured simultaneously
        while the same hand is visible to both devices.

        Returns 4x4 SE3 matrix. Uses Kabsch algorithm (SVD) on matched joint
        positions.
        """
        pts_a, pts_b = [], []
        for fa, fb in observations:
            if not fa.poses or not fb.poses:
                continue
            pa, pb = fa.poses[0], fb.poses[0]
            if not pa.joints or not pb.joints:
                continue
            for ja, jb in zip(pa.joints, pb.joints):
                if ja.valid and jb.valid:
                    pts_a.append(ja.position)
                    pts_b.append(jb.position)

        if len(pts_a) < 3:
            return np.eye(4)

        A = np.array(pts_a)
        B = np.array(pts_b)
        ca, cb = A.mean(axis=0), B.mean(axis=0)
        A_c, B_c = A - ca, B - cb

        H = B_c.T @ A_c
        U, _, Vt = np.linalg.svd(H)
        d = np.linalg.det(Vt.T @ U.T)
        S = np.diag([1, 1, d])
        R = Vt.T @ S @ U.T
        t = ca - R @ cb

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T


def default_config(topology: str, device_ids: list[str]) -> FusionConfig:
    """Create a FusionConfig with reasonable defaults for a topology."""
    return FusionConfig(
        topology=topology,
        device_ids=list(device_ids),
        weights={did: 1.0 for did in device_ids},
    )


if __name__ == "__main__":
    j_a = tuple(Joint(position=(float(i) * 0.01, 0.1, -0.3), valid=True)
                for i in range(26))
    j_b = tuple(Joint(position=(float(i) * 0.01 + 0.002, 0.102, -0.298), valid=True)
                for i in range(26))
    fa = PoseFrame(seq=1, source_id="dev-a", poses=(Pose(joints=j_a, confidence=0.9),))
    fb = PoseFrame(seq=1, source_id="dev-b", poses=(Pose(joints=j_b, confidence=0.8),))

    cfg = default_config(DUAL, ["dev-a", "dev-b"])
    engine = FusionEngine(cfg)
    now = time.monotonic()
    engine.add_frame("dev-a", fa, now)
    engine.add_frame("dev-b", fb, now)
    fused = engine.get_fused_frame()

    if fused and fused.poses:
        p = fused.poses[0]
        print(f"Fused: {len(p.joints)} joints, conf={p.confidence:.2f}")
        print(f"  palm={p.joints[0].position}")
        print(f"  wrist={p.joints[1].position}")
    else:
        print("Fusion produced no output")

    T = FusionEngine.calibrate_pair([(fa, fb)])
    print(f"\nCalibration (A←B):\n  translation={T[:3, 3]}")
    print(f"  rotation diagonal={np.diag(T[:3, :3])}")
    print("Self-check passed." if fused else "FAILED")
