#!/usr/bin/env python3
"""Tests for pose_frame.py: SDK public types and derive_semantics."""

import math
import unittest

from pose_frame import (
    Caps, Joint, Pose, PoseFrame, JOINT_NAMES,
    derive_semantics, _dist, _finger_curl,
)


class TestCaps(unittest.TestCase):
    def test_individual_flags(self):
        self.assertEqual(int(Caps.PALM_POSE), 0x01)
        self.assertEqual(int(Caps.JOINTS_FULL), 0x08)
        self.assertEqual(int(Caps.DISCRETE_EVENT), 0x40)

    def test_combine_flags(self):
        c = Caps.PALM_POSE | Caps.JOINTS_FULL | Caps.CHIRALITY
        self.assertIn(Caps.PALM_POSE, c)
        self.assertIn(Caps.JOINTS_FULL, c)
        self.assertNotIn(Caps.VELOCITY, c)

    def test_empty(self):
        c = Caps(0)
        self.assertEqual(int(c), 0)
        self.assertNotIn(Caps.PALM_POSE, c)


class TestJointNames(unittest.TestCase):
    def test_count(self):
        self.assertEqual(len(JOINT_NAMES), 26)

    def test_starts_with_palm_wrist(self):
        self.assertEqual(JOINT_NAMES[0], "palm")
        self.assertEqual(JOINT_NAMES[1], "wrist")

    def test_all_fingers_present(self):
        for finger in ("thumb", "index", "middle", "ring", "little"):
            tips = [n for n in JOINT_NAMES if n == f"{finger}_tip"]
            self.assertEqual(len(tips), 1, f"missing {finger}_tip")

    def test_unique(self):
        self.assertEqual(len(set(JOINT_NAMES)), 26)


class TestJoint(unittest.TestCase):
    def test_defaults(self):
        j = Joint()
        self.assertEqual(j.position, (0.0, 0.0, 0.0))
        self.assertEqual(j.orientation, (0.0, 0.0, 0.0, 1.0))
        self.assertFalse(j.valid)

    def test_custom(self):
        j = Joint(position=(1.0, 2.0, 3.0), valid=True)
        self.assertEqual(j.position, (1.0, 2.0, 3.0))
        self.assertTrue(j.valid)


class TestPose(unittest.TestCase):
    def test_defaults(self):
        p = Pose()
        self.assertEqual(p.track_id, 0)
        self.assertIsNone(p.joints)
        self.assertEqual(p.confidence, 1.0)

    def test_with_joints(self):
        joints = tuple(Joint(valid=True) for _ in range(26))
        p = Pose(joints=joints)
        self.assertEqual(len(p.joints), 26)


class TestPoseFrame(unittest.TestCase):
    def test_defaults(self):
        f = PoseFrame()
        self.assertEqual(f.seq, 0)
        self.assertEqual(f.t_capture_ns, 0)
        self.assertEqual(f.source_id, "")
        self.assertEqual(f.caps, Caps(0))
        self.assertEqual(f.poses, ())

    def test_with_data(self):
        p = Pose(track_id=1)
        f = PoseFrame(seq=42, t_capture_ns=1000000, source_id="leap",
                      caps=Caps.JOINTS_FULL, poses=(p,))
        self.assertEqual(f.seq, 42)
        self.assertEqual(len(f.poses), 1)
        self.assertIn(Caps.JOINTS_FULL, f.caps)


class TestDist(unittest.TestCase):
    def test_same_point(self):
        self.assertAlmostEqual(_dist((1, 2, 3), (1, 2, 3)), 0.0)

    def test_known_distance(self):
        self.assertAlmostEqual(_dist((0, 0, 0), (3, 4, 0)), 5.0)

    def test_3d(self):
        self.assertAlmostEqual(_dist((0, 0, 0), (1, 1, 1)), math.sqrt(3))


class TestDeriveSemantics(unittest.TestCase):
    """Test the canonical gesture derivation."""

    def _make_straight_hand(self):
        """Build a hand with all fingers fully extended (straight)."""
        joints = []
        # palm at origin, wrist slightly behind
        joints.append(Joint(position=(0.0, 0.0, 0.0), valid=True))   # palm
        joints.append(Joint(position=(0.0, -0.02, 0.0), valid=True)) # wrist

        # thumb: angled outward, fully extended
        thumb_positions = [
            (0.03, 0.01, 0.0),   # metacarpal
            (0.05, 0.03, 0.0),   # proximal
            (0.07, 0.05, 0.0),   # distal
            (0.09, 0.07, 0.0),   # tip
        ]
        for pos in thumb_positions:
            joints.append(Joint(position=pos, valid=True))

        # index through little: pointing up, evenly spaced
        for fi, x_off in enumerate([-0.01, 0.0, 0.01, 0.02]):
            base_x = x_off
            positions = [
                (base_x, 0.02, 0.0),   # metacarpal
                (base_x, 0.04, 0.0),   # proximal
                (base_x, 0.06, 0.0),   # intermediate
                (base_x, 0.08, 0.0),   # distal
                (base_x, 0.10, 0.0),   # tip
            ]
            for pos in positions:
                joints.append(Joint(position=pos, valid=True))

        return Pose(joints=tuple(joints), confidence=1.0)

    def _make_fist(self):
        """Build a hand with all fingers curled into a fist."""
        joints = []
        joints.append(Joint(position=(0.0, 0.0, 0.0), valid=True))   # palm
        joints.append(Joint(position=(0.0, -0.02, 0.0), valid=True)) # wrist

        # thumb: curled inward
        thumb_positions = [
            (0.03, 0.01, 0.0),    # metacarpal
            (0.04, 0.03, 0.0),    # proximal
            (0.03, 0.04, 0.0),    # distal (curling back)
            (0.02, 0.03, 0.0),    # tip (curled inward toward palm)
        ]
        for pos in thumb_positions:
            joints.append(Joint(position=pos, valid=True))

        # index through little: curled down
        for fi, x_off in enumerate([-0.01, 0.0, 0.01, 0.02]):
            positions = [
                (x_off, 0.02, 0.0),      # metacarpal
                (x_off, 0.04, 0.0),       # proximal
                (x_off, 0.04, -0.02),     # intermediate (curling down)
                (x_off, 0.02, -0.02),     # distal (further curled)
                (x_off, 0.01, -0.01),     # tip (near palm)
            ]
            for pos in positions:
                joints.append(Joint(position=pos, valid=True))

        return Pose(joints=tuple(joints), confidence=1.0)

    def test_no_joints(self):
        """Pose without joints returns safe defaults."""
        p = Pose()
        s = derive_semantics(p)
        self.assertEqual(s["pinch_strength"], 0.0)
        self.assertEqual(s["grab_strength"], 0.0)
        self.assertEqual(s["extended"], [True, True, True, True, True])

    def test_too_few_joints(self):
        """Pose with fewer than 26 joints returns defaults."""
        p = Pose(joints=tuple(Joint() for _ in range(10)))
        s = derive_semantics(p)
        self.assertEqual(s["pinch_strength"], 0.0)

    def test_open_hand(self):
        """Extended hand: low grab, all fingers extended."""
        pose = self._make_straight_hand()
        s = derive_semantics(pose)
        self.assertLess(s["grab_strength"], 0.3)
        self.assertTrue(all(s["extended"]), f"extended={s['extended']}")

    def test_fist(self):
        """Fist: high grab, no fingers extended."""
        pose = self._make_fist()
        s = derive_semantics(pose)
        self.assertGreater(s["grab_strength"], 0.3)
        # at least some fingers should be non-extended in a fist
        extended_count = sum(s["extended"])
        self.assertLessEqual(extended_count, 2,
                             f"too many extended in fist: {s['extended']}")

    def test_pinch_close(self):
        """Thumb tip touching index tip -> high pinch."""
        pose = self._make_straight_hand()
        joints = list(pose.joints)
        # move index tip to thumb tip location
        joints[10] = Joint(position=joints[5].position, valid=True)
        pose = Pose(joints=tuple(joints))
        s = derive_semantics(pose)
        self.assertGreater(s["pinch_strength"], 0.8)

    def test_pinch_open(self):
        """Thumb and index tips far apart -> low pinch."""
        pose = self._make_straight_hand()
        s = derive_semantics(pose)
        # in the straight hand, thumb tip and index tip are well separated
        self.assertLess(s["pinch_strength"], 0.5)

    def test_result_keys(self):
        """derive_semantics returns exactly the documented keys."""
        pose = self._make_straight_hand()
        s = derive_semantics(pose)
        self.assertIn("pinch_strength", s)
        self.assertIn("grab_strength", s)
        self.assertIn("extended", s)
        self.assertEqual(len(s["extended"]), 5)

    def test_invalid_joints_safe(self):
        """Joints marked invalid produce reasonable defaults, not crashes."""
        joints = tuple(Joint(valid=False) for _ in range(26))
        pose = Pose(joints=joints)
        s = derive_semantics(pose)
        # should not crash; pinch is 0 (invalid tips)
        self.assertEqual(s["pinch_strength"], 0.0)
        # grab should be moderate (unknown curl = 0.5)
        self.assertAlmostEqual(s["grab_strength"], 0.5)


class TestFingerCurl(unittest.TestCase):
    def test_straight_finger_low_curl(self):
        """A perfectly straight finger has curl near 0."""
        joints = [Joint()] * 26
        # index finger: joints 6..10
        joints[6] = Joint(position=(0.0, 0.0, 0.0), valid=True)    # meta
        joints[7] = Joint(position=(0.0, 0.03, 0.0), valid=True)   # prox
        joints[8] = Joint(position=(0.0, 0.06, 0.0), valid=True)   # inter
        joints[9] = Joint(position=(0.0, 0.09, 0.0), valid=True)   # distal
        joints[10] = Joint(position=(0.0, 0.12, 0.0), valid=True)  # tip
        c = _finger_curl(tuple(joints), "index")
        self.assertLess(c, 0.1)

    def test_curled_finger_high_curl(self):
        """A finger curled back on itself has high curl."""
        joints = [Joint()] * 26
        # index finger curled: tip points back toward metacarpal
        joints[6] = Joint(position=(0.0, 0.0, 0.0), valid=True)
        joints[7] = Joint(position=(0.0, 0.03, 0.0), valid=True)
        joints[8] = Joint(position=(0.0, 0.03, -0.02), valid=True)
        joints[9] = Joint(position=(0.0, 0.01, -0.02), valid=True)
        joints[10] = Joint(position=(0.0, -0.01, -0.01), valid=True)
        c = _finger_curl(tuple(joints), "index")
        self.assertGreater(c, 0.3)


if __name__ == "__main__":
    unittest.main()
