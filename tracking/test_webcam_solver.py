#!/usr/bin/env python3
"""Tests for WebcamPoseSolver with mock ONNX sessions."""

import logging
import time
import unittest

import numpy as np

from webcam_solver import (
    ANCHORS,
    DETECTION_THRESHOLD,
    LANDMARK_SIZE,
    NUM_ANCHORS,
    NUM_LANDMARKS,
    NUM_PALM_KEYPOINTS,
    PALM_SIZE,
    Hand,
    WebcamPoseSolver,
    _bilinear_resize,
    _generate_anchors,
    _iou,
    _nms,
    _sigmoid,
)


# ---------------------------------------------------------------------------
# Mock ONNX session
# ---------------------------------------------------------------------------
class MockSession:
    """Mimics onnxruntime.InferenceSession for testing."""

    def __init__(self, outputs_fn):
        """outputs_fn(input_dict) -> list of ndarrays."""
        self._fn = outputs_fn

    def run(self, output_names, input_dict):
        return self._fn(input_dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_gray_bytes(width, height, value=128):
    """Create a uniform grayscale frame."""
    return bytes([value] * (width * height))


def _palm_no_detection(input_dict):
    """Palm detector that returns zero scores (no detections)."""
    return [
        np.zeros((1, NUM_ANCHORS, 18), dtype=np.float32),
        np.full((1, NUM_ANCHORS, 1), -10.0, dtype=np.float32),  # sigmoid -> ~0
    ]


def _palm_one_detection(input_dict):
    """Palm detector that fires anchor 0 with a high score and a centred box."""
    regs = np.zeros((1, NUM_ANCHORS, 18), dtype=np.float32)
    scores = np.full((1, NUM_ANCHORS, 1), -10.0, dtype=np.float32)
    # anchor 0 sits at roughly (0.03, 0.03) for stride-8 grid
    # give it a moderate offset so the decoded box lands near the centre
    anch_cx, anch_cy = float(ANCHORS[0, 0]), float(ANCHORS[0, 1])
    target_cx, target_cy = 0.5, 0.5
    regs[0, 0, 0] = (target_cx - anch_cx) * PALM_SIZE
    regs[0, 0, 1] = (target_cy - anch_cy) * PALM_SIZE
    regs[0, 0, 2] = 0.3 * PALM_SIZE   # width
    regs[0, 0, 3] = 0.3 * PALM_SIZE   # height
    scores[0, 0, 0] = 5.0  # sigmoid -> ~0.993
    return [regs, scores]


def _landmark_valid(input_dict):
    """Landmark model that returns evenly-spaced joints and high presence."""
    lm = np.zeros((1, 63), dtype=np.float32)
    for j in range(NUM_LANDMARKS):
        lm[0, j * 3 + 0] = (j / NUM_LANDMARKS) * LANDMARK_SIZE  # x
        lm[0, j * 3 + 1] = (j / NUM_LANDMARKS) * LANDMARK_SIZE  # y
        lm[0, j * 3 + 2] = 0.0                                   # z
    handedness = np.array([[2.0]], dtype=np.float32)   # sigmoid -> ~0.88
    presence = np.array([[5.0]], dtype=np.float32)     # sigmoid -> ~0.993
    return [lm, handedness, presence]


def _landmark_low_presence(input_dict):
    """Landmark model that returns low presence (hand not really there)."""
    lm = np.zeros((1, 63), dtype=np.float32)
    handedness = np.array([[0.0]], dtype=np.float32)
    presence = np.array([[-5.0]], dtype=np.float32)   # sigmoid -> ~0.007
    return [lm, handedness, presence]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestAnchors(unittest.TestCase):

    def test_anchor_count(self):
        anchors = _generate_anchors()
        self.assertEqual(anchors.shape, (NUM_ANCHORS, 2))

    def test_anchor_range(self):
        self.assertTrue(np.all(ANCHORS >= 0.0))
        self.assertTrue(np.all(ANCHORS <= 1.0))

    def test_anchor_stride_breakdown(self):
        """Verify anchor counts per feature-map layer."""
        # stride 8 -> 16x16 grid, 2 anchors/cell = 512
        # stride 16 (x3) -> 8x8 grid, 2 anchors/cell = 128 each = 384
        self.assertEqual(512 + 384, NUM_ANCHORS)

    def test_first_anchor_position(self):
        # stride=8, grid=16, first cell (0,0) -> cx = 0.5/16
        expected = 0.5 / 16
        np.testing.assert_almost_equal(ANCHORS[0, 0], expected)
        np.testing.assert_almost_equal(ANCHORS[0, 1], expected)


class TestSigmoid(unittest.TestCase):

    def test_zero(self):
        np.testing.assert_almost_equal(_sigmoid(np.array([0.0])), [0.5])

    def test_large_positive(self):
        result = _sigmoid(np.array([100.0]))
        self.assertAlmostEqual(float(result[0]), 1.0, places=5)

    def test_large_negative(self):
        result = _sigmoid(np.array([-100.0]))
        self.assertAlmostEqual(float(result[0]), 0.0, places=5)

    def test_symmetry(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        s = _sigmoid(x)
        np.testing.assert_array_almost_equal(s + s[::-1],
                                             np.ones(5), decimal=6)


class TestBilinearResize(unittest.TestCase):

    def test_identity(self):
        img = np.arange(25, dtype=np.uint8).reshape(5, 5)
        out = _bilinear_resize(img, 5, 5)
        np.testing.assert_array_almost_equal(out, img.astype(np.float32))

    def test_downsample(self):
        img = np.ones((100, 100), dtype=np.uint8) * 200
        out = _bilinear_resize(img, 10, 10)
        self.assertEqual(out.shape, (10, 10))
        np.testing.assert_array_almost_equal(out, 200.0, decimal=0)

    def test_upsample(self):
        img = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        out = _bilinear_resize(img, 4, 4)
        self.assertEqual(out.shape, (4, 4))
        # corners should lean toward their source value
        self.assertLess(out[0, 0], 128)
        self.assertGreater(out[0, 3], 128)

    def test_single_pixel(self):
        img = np.array([[42]], dtype=np.uint8)
        out = _bilinear_resize(img, 8, 8)
        np.testing.assert_array_almost_equal(out, 42.0, decimal=0)


class TestIoU(unittest.TestCase):

    def test_identical(self):
        box = [0.0, 0.0, 1.0, 1.0]
        self.assertAlmostEqual(_iou(box, box), 1.0)

    def test_no_overlap(self):
        a = [0.0, 0.0, 0.5, 0.5]
        b = [0.6, 0.6, 1.0, 1.0]
        self.assertAlmostEqual(_iou(a, b), 0.0)

    def test_partial(self):
        a = [0.0, 0.0, 0.5, 0.5]
        b = [0.25, 0.25, 0.75, 0.75]
        iou = _iou(a, b)
        self.assertGreater(iou, 0.0)
        self.assertLess(iou, 1.0)


class TestNMS(unittest.TestCase):

    def test_single(self):
        boxes = np.array([[0.0, 0.0, 1.0, 1.0]])
        scores = np.array([0.9])
        keep = _nms(boxes, scores)
        self.assertEqual(keep, [0])

    def test_overlapping_suppressed(self):
        boxes = np.array([
            [0.0, 0.0, 0.5, 0.5],
            [0.05, 0.05, 0.55, 0.55],  # high overlap with first
        ])
        scores = np.array([0.9, 0.8])
        keep = _nms(boxes, scores, iou_threshold=0.3)
        self.assertEqual(keep, [0])

    def test_separate_kept(self):
        boxes = np.array([
            [0.0, 0.0, 0.3, 0.3],
            [0.7, 0.7, 1.0, 1.0],
        ])
        scores = np.array([0.9, 0.8])
        keep = _nms(boxes, scores)
        self.assertEqual(sorted(keep), [0, 1])


class TestWebcamPoseSolverNoDetection(unittest.TestCase):

    def test_empty_frame(self):
        palm = MockSession(_palm_no_detection)
        lm = MockSession(_landmark_valid)
        solver = WebcamPoseSolver._from_sessions(palm, lm)
        gray = _make_gray_bytes(320, 240)
        hands = solver(gray, 320, 240)
        self.assertEqual(hands, [])


class TestWebcamPoseSolverOneHand(unittest.TestCase):

    def setUp(self):
        palm = MockSession(_palm_one_detection)
        lm = MockSession(_landmark_valid)
        self.solver = WebcamPoseSolver._from_sessions(palm, lm)

    def test_returns_one_hand(self):
        gray = _make_gray_bytes(320, 240)
        hands = self.solver(gray, 320, 240)
        self.assertEqual(len(hands), 1)

    def test_hand_fields(self):
        gray = _make_gray_bytes(320, 240)
        hands = self.solver(gray, 320, 240)
        h = hands[0]
        self.assertIsInstance(h, Hand)
        self.assertEqual(h.landmarks.shape, (NUM_LANDMARKS, 3))
        self.assertGreater(h.confidence, 0.0)
        self.assertEqual(len(h.bbox), 4)
        self.assertGreaterEqual(h.handedness, 0.0)
        self.assertLessEqual(h.handedness, 1.0)

    def test_landmarks_in_frame_coords(self):
        w, h = 640, 480
        gray = _make_gray_bytes(w, h)
        hands = self.solver(gray, w, h)
        lm = hands[0].landmarks
        # landmarks should be mapped into frame pixel space -- at least
        # some should be within bounds (they come from the crop mapping)
        x_vals = lm[:, 0]
        y_vals = lm[:, 1]
        # the box is centred at ~(0.5, 0.5) of the frame, so landmarks
        # should cluster around the centre +-crop margin
        self.assertTrue(np.any((x_vals > 0) & (x_vals < w)))
        self.assertTrue(np.any((y_vals > 0) & (y_vals < h)))

    def test_bbox_in_pixel_coords(self):
        w, h = 320, 240
        gray = _make_gray_bytes(w, h)
        hands = self.solver(gray, w, h)
        x0, y0, x1, y1 = hands[0].bbox
        self.assertGreaterEqual(x0, 0)
        self.assertGreaterEqual(y0, 0)
        self.assertLessEqual(x1, w)
        self.assertLessEqual(y1, h)


class TestWebcamPoseSolverLowPresence(unittest.TestCase):

    def test_filtered_out(self):
        """Landmark model returns low presence -> hand is discarded."""
        palm = MockSession(_palm_one_detection)
        lm = MockSession(_landmark_low_presence)
        solver = WebcamPoseSolver._from_sessions(palm, lm)
        gray = _make_gray_bytes(320, 240)
        hands = solver(gray, 320, 240)
        self.assertEqual(hands, [])


class TestFPSLogging(unittest.TestCase):

    def test_fps_logged_after_interval(self):
        palm = MockSession(_palm_no_detection)
        lm = MockSession(_landmark_valid)
        solver = WebcamPoseSolver._from_sessions(palm, lm)

        # force the start time into the past so the next call triggers logging
        solver._fps_start = time.monotonic() - 10.0
        solver._frame_count = 49

        with self.assertLogs("webcam_solver", level="INFO") as cm:
            gray = _make_gray_bytes(64, 64)
            solver(gray, 64, 64)

        self.assertTrue(any("FPS" in msg for msg in cm.output))


class TestPreprocessPalm(unittest.TestCase):

    def test_output_shape(self):
        palm = MockSession(_palm_no_detection)
        lm = MockSession(_landmark_valid)
        solver = WebcamPoseSolver._from_sessions(palm, lm)
        gray = np.zeros((240, 320), dtype=np.uint8)
        out = solver._preprocess_palm(gray)
        self.assertEqual(out.shape, (1, 3, PALM_SIZE, PALM_SIZE))

    def test_output_range(self):
        palm = MockSession(_palm_no_detection)
        lm = MockSession(_landmark_valid)
        solver = WebcamPoseSolver._from_sessions(palm, lm)
        gray = np.random.randint(0, 256, (120, 160), dtype=np.uint8)
        out = solver._preprocess_palm(gray)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)

    def test_three_channels_identical(self):
        """Pseudo-RGB: all three channels should be identical."""
        palm = MockSession(_palm_no_detection)
        lm = MockSession(_landmark_valid)
        solver = WebcamPoseSolver._from_sessions(palm, lm)
        gray = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        out = solver._preprocess_palm(gray)
        np.testing.assert_array_equal(out[0, 0], out[0, 1])
        np.testing.assert_array_equal(out[0, 0], out[0, 2])


if __name__ == "__main__":
    unittest.main()
