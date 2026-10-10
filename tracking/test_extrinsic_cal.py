#!/usr/bin/env python3
"""Tests for extrinsic_calibration.py: save/load, transform, solve with
synthetic data."""

import os
import tempfile
import unittest

import cv2
import numpy as np

from extrinsic_calibration import CalibrationResult, CalibrationManager


class TestCalibrationResult(unittest.TestCase):
    """Unit tests for CalibrationResult dataclass."""

    def _make_sample(self):
        """A known rotation (90 deg about Z) and translation."""
        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        t = np.array([[100], [200], [300]], dtype=np.float64)
        return CalibrationResult(rotation=R, translation=t,
                                 reprojection_error=1.5)

    def test_transform_point(self):
        cal = self._make_sample()
        # (1,0,0) rotated 90 deg about Z -> (0,1,0), then + (100,200,300)
        out = cal.transform_point(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(out, [100.0, 201.0, 300.0], atol=1e-10)

    def test_transform_point_origin(self):
        cal = self._make_sample()
        out = cal.transform_point(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(out, [100.0, 200.0, 300.0], atol=1e-10)

    def test_inverse_roundtrip(self):
        cal = self._make_sample()
        inv = cal.inverse()
        p = np.array([5.0, -3.0, 7.0])
        fwd = cal.transform_point(p)
        back = inv.transform_point(fwd)
        np.testing.assert_allclose(back, p, atol=1e-10)

    def test_inverse_rotation_is_transpose(self):
        cal = self._make_sample()
        inv = cal.inverse()
        np.testing.assert_allclose(inv.rotation, cal.rotation.T, atol=1e-12)

    def test_save_load_roundtrip(self):
        cal = self._make_sample()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            cal.save(path)
            loaded = CalibrationResult.load(path)
            np.testing.assert_allclose(loaded.rotation, cal.rotation)
            np.testing.assert_allclose(loaded.translation, cal.translation)
            self.assertAlmostEqual(loaded.reprojection_error,
                                   cal.reprojection_error, places=6)
        finally:
            os.unlink(path)

    def test_save_creates_file(self):
        cal = self._make_sample()
        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            cal.save(path)
            self.assertTrue(os.path.exists(path))
            d = np.load(path)
            self.assertIn("R", d)
            self.assertIn("t", d)
            self.assertIn("err", d)
        finally:
            os.unlink(path)


class TestCalibrationManagerSyntheticSolve(unittest.TestCase):
    """Solve with synthetic 3D-2D data from a known ground-truth transform."""

    def setUp(self):
        """Set up a synthetic camera and a known Leap->Webcam transform."""
        # simple pinhole camera
        self.fx, self.fy = 600.0, 600.0
        self.cx, self.cy = 320.0, 240.0
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)

        # ground truth: Rx(-90 deg), translation (0, 250, -150) mm
        # (Leap looking up, webcam above and behind, looking forward)
        angle = -np.pi / 2
        self.R_true = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ], dtype=np.float64)
        self.t_true = np.array([[0], [250], [-150]], dtype=np.float64)

    def _project(self, pts_3d):
        """Project 3D Leap points into the webcam image via the ground truth."""
        rvec, _ = cv2.Rodrigues(self.R_true)
        proj, _ = cv2.projectPoints(pts_3d, rvec, self.t_true, self.K, None)
        return proj.reshape(-1, 2)

    def _generate_correspondences(self, n=30, rng=None):
        """Generate n random 3D points in Leap space and project them."""
        if rng is None:
            rng = np.random.default_rng(42)
        # points in a box roughly where hands appear: x in [-100,100],
        # y in [100,400], z in [-200,-50] (mm, Leap frame)
        pts_3d = np.column_stack([
            rng.uniform(-100, 100, n),
            rng.uniform(100, 400, n),
            rng.uniform(-200, -50, n),
        ])
        pts_2d = self._project(pts_3d)
        return pts_3d, pts_2d

    def test_solve_recovers_transform(self):
        """solvePnP with clean synthetic data recovers the ground truth."""
        pts_3d, pts_2d = self._generate_correspondences(40)
        mgr = CalibrationManager(self.K)
        mgr.add_correspondence(pts_3d, pts_2d)
        result = mgr.solve()

        # rotation should be close to ground truth
        np.testing.assert_allclose(result.rotation, self.R_true, atol=0.05)
        # translation within a few mm
        np.testing.assert_allclose(result.translation.flatten(),
                                   self.t_true.flatten(), atol=5.0)
        # reprojection error should be near zero for clean data
        self.assertLess(result.reprojection_error, 1.0)

    def test_solve_with_noise(self):
        """Solve tolerates moderate 2D noise."""
        rng = np.random.default_rng(99)
        pts_3d, pts_2d = self._generate_correspondences(50, rng)
        # add ~2 px Gaussian noise to the 2D observations
        pts_2d += rng.normal(0, 2.0, pts_2d.shape)
        mgr = CalibrationManager(self.K)
        mgr.add_correspondence(pts_3d, pts_2d)
        result = mgr.solve()

        # looser bounds due to noise
        np.testing.assert_allclose(result.rotation, self.R_true, atol=0.15)
        np.testing.assert_allclose(result.translation.flatten(),
                                   self.t_true.flatten(), atol=30.0)
        self.assertLess(result.reprojection_error, 5.0)

    def test_too_few_points_raises(self):
        mgr = CalibrationManager(self.K)
        mgr.add_correspondence(np.array([1, 2, 3]), np.array([10, 20]))
        with self.assertRaises(ValueError):
            mgr.solve()

    def test_add_single_and_batch(self):
        """Both single-point and batch add work."""
        mgr = CalibrationManager(self.K)
        # single
        mgr.add_correspondence(np.array([1, 2, 3]), np.array([10, 20]))
        self.assertEqual(mgr.num_correspondences, 1)
        # batch of 5
        mgr.add_correspondence(np.ones((5, 3)), np.ones((5, 2)))
        self.assertEqual(mgr.num_correspondences, 6)

    def test_mismatched_counts_raises(self):
        mgr = CalibrationManager(self.K)
        with self.assertRaises(ValueError):
            mgr.add_correspondence(np.ones((3, 3)), np.ones((2, 2)))


class TestCheckDrift(unittest.TestCase):
    """Test the drift-detection method."""

    def setUp(self):
        self.K = np.array([
            [600, 0, 320],
            [0, 600, 240],
            [0, 0, 1],
        ], dtype=np.float64)

    def test_no_drift(self):
        """Points projected through the calibration and observed identically
        should show no drift."""
        R = np.eye(3)
        t = np.array([[0], [0], [500]], dtype=np.float64)
        cal = CalibrationResult(rotation=R, translation=t,
                                reprojection_error=0.5)
        # project a point through the calibration
        p3 = np.array([[10, 20, 0]], dtype=np.float64)
        rvec, _ = cv2.Rodrigues(R)
        p2, _ = cv2.projectPoints(p3, rvec, t, self.K, None)
        p2 = p2.reshape(-1, 2)

        mgr = CalibrationManager(self.K)
        self.assertFalse(mgr.check_drift(p3, p2, cal, threshold_px=1.0))

    def test_large_drift(self):
        """Observations far from projected locations should trigger drift."""
        R = np.eye(3)
        t = np.array([[0], [0], [500]], dtype=np.float64)
        cal = CalibrationResult(rotation=R, translation=t,
                                reprojection_error=0.5)
        p3 = np.array([[10, 20, 0]], dtype=np.float64)
        # deliberately wrong 2D observation
        p2 = np.array([[100, 100]], dtype=np.float64)

        mgr = CalibrationManager(self.K)
        self.assertTrue(mgr.check_drift(p3, p2, cal, threshold_px=5.0))


if __name__ == "__main__":
    unittest.main()
