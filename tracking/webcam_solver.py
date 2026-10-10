#!/usr/bin/env python3
"""WebcamPoseSolver -- detect hands in grayscale frames using MediaPipe ONNX
models (palm detection + hand landmarks).  Pure numpy, no opencv or mediapipe
pip packages.

Models:
  palm_detection_MEDIAPIPE.onnx   IN: [1,3,128,128]  OUT: regressors[1,896,18], classificators[1,896,1]
  hand_landmark_MEDIAPIPE.onnx    IN: [1,3,224,224]  OUT: landmarks[1,63], handedness[1,1], presence[1,1]

The palm detector is a BlazePalm SSD with 896 anchors across four feature-map
layers (strides 8,16,16,16, two anchors per cell).  Each anchor regresses
[cx, cy, w, h, kp0_x, kp0_y, ..., kp6_x, kp6_y] offsets scaled by the input
size (128).  Scores are raw logits passed through sigmoid.

The landmark model outputs 21 joints x 3 coords (x, y, z) in the 224x224 crop
coordinate space, plus handedness and presence logits.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PALM_SIZE = 128       # palm_detection_MEDIAPIPE input edge
LANDMARK_SIZE = 224   # hand_landmark_MEDIAPIPE input edge
NUM_ANCHORS = 896
DETECTION_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.3
LANDMARK_PRESENCE_THRESHOLD = 0.5
NUM_PALM_KEYPOINTS = 7   # keypoints emitted by the palm detector
NUM_LANDMARKS = 21        # hand-skeleton joints
CROP_EXPAND = 1.4         # expand the palm box before landmark crop

# MediaPipe 21-joint naming (matches the landmark model's output order)
JOINT_NAMES = [
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass
class Hand:
    """A detected hand with 21 landmarks in frame-pixel coordinates."""
    landmarks: np.ndarray   # [21, 3] float32 -- x, y in frame px; z relative
    confidence: float       # combined palm-score * landmark-presence
    bbox: tuple             # (x0, y0, x1, y1) in frame pixels
    handedness: float       # 0.0 = left, 1.0 = right (after sigmoid)


# ---------------------------------------------------------------------------
# Anchor generation (matches MediaPipe SsdAnchorsCalculator config)
# ---------------------------------------------------------------------------
def _generate_anchors():
    """Generate the 896 SSD anchors for the 128x128 palm detection model.

    Config: strides=[8,16,16,16], 2 anchors per cell, anchor_offset=0.5,
    fixed_anchor_size=True.  Each anchor is [cx, cy] normalised to [0, 1].
    """
    anchors = []
    strides = [8, 16, 16, 16]
    for stride in strides:
        grid = PALM_SIZE // stride
        for y in range(grid):
            for x in range(grid):
                cx = (x + 0.5) / grid
                cy = (y + 0.5) / grid
                # two anchors per cell (aspect_ratios=[1.0], two scales)
                anchors.append([cx, cy])
                anchors.append([cx, cy])
    return np.array(anchors, dtype=np.float32)


ANCHORS = _generate_anchors()
assert ANCHORS.shape == (NUM_ANCHORS, 2), (
    f"anchor count mismatch: expected {NUM_ANCHORS}, got {ANCHORS.shape[0]}")


# ---------------------------------------------------------------------------
# Numpy-only helpers
# ---------------------------------------------------------------------------
def _sigmoid(x):
    """Numerically stable sigmoid."""
    pos = x >= 0
    out = np.empty_like(x, dtype=np.float32)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def _bilinear_resize(img, out_h, out_w):
    """Resize a 2-D array (H, W) via bilinear interpolation.  Numpy-only."""
    in_h, in_w = img.shape
    if in_h == out_h and in_w == out_w:
        return img.astype(np.float32)

    y_scale = in_h / out_h
    x_scale = in_w / out_w
    # output pixel centres mapped back to input space
    oy = (np.arange(out_h, dtype=np.float64) + 0.5) * y_scale - 0.5
    ox = (np.arange(out_w, dtype=np.float64) + 0.5) * x_scale - 0.5
    oy = np.clip(oy, 0, in_h - 1.0001)
    ox = np.clip(ox, 0, in_w - 1.0001)

    y0 = np.floor(oy).astype(np.intp)
    x0 = np.floor(ox).astype(np.intp)
    y1 = np.minimum(y0 + 1, in_h - 1)
    x1 = np.minimum(x0 + 1, in_w - 1)
    ya = (oy - y0).astype(np.float32)[:, None]   # [out_h, 1]
    xa = (ox - x0).astype(np.float32)[None, :]   # [1, out_w]

    f = img.astype(np.float32)
    tl = f[y0[:, None], x0[None, :]]
    tr = f[y0[:, None], x1[None, :]]
    bl = f[y1[:, None], x0[None, :]]
    br = f[y1[:, None], x1[None, :]]
    return tl * (1 - xa) * (1 - ya) + tr * xa * (1 - ya) + \
        bl * (1 - xa) * ya + br * xa * ya


def _iou(a, b):
    """IoU between two [x0, y0, x1, y1] boxes."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def _nms(boxes, scores, iou_threshold=NMS_IOU_THRESHOLD):
    """Greedy non-maximum suppression.  Returns kept indices."""
    order = np.argsort(-scores)
    keep = []
    suppressed = set()
    for rank, i in enumerate(order):
        if i in suppressed:
            continue
        keep.append(int(i))
        for j in order[rank + 1:]:
            if j not in suppressed and _iou(boxes[i], boxes[j]) > iou_threshold:
                suppressed.add(int(j))
    return keep


# ---------------------------------------------------------------------------
# WebcamPoseSolver
# ---------------------------------------------------------------------------
class WebcamPoseSolver:
    """Detect hands in grayscale frames via MediaPipe ONNX models.

    Uses ``palm_detection_MEDIAPIPE.onnx`` for SSD-based palm detection and
    ``hand_landmark_MEDIAPIPE.onnx`` for 21-joint landmark regression.
    Requires only numpy and onnxruntime.

    Parameters
    ----------
    models_dir : str or None
        Path to the directory containing the two ``.onnx`` files.  Defaults to
        ``../hand-tracking-models/`` relative to this source file.
    """

    def __init__(self, models_dir: Optional[str] = None):
        import onnxruntime as ort   # deferred so tests can mock it

        if models_dir is None:
            models_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "hand-tracking-models")

        palm_path = os.path.join(models_dir, "palm_detection_MEDIAPIPE.onnx")
        lm_path = os.path.join(models_dir, "hand_landmark_MEDIAPIPE.onnx")
        if not os.path.isfile(palm_path):
            raise FileNotFoundError(f"palm detection model not found: {palm_path}")
        if not os.path.isfile(lm_path):
            raise FileNotFoundError(f"hand landmark model not found: {lm_path}")

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self._palm_session = ort.InferenceSession(palm_path, opts)
        self._landmark_session = ort.InferenceSession(lm_path, opts)

        self._anchors = ANCHORS

        # FPS bookkeeping
        self._frame_count = 0
        self._fps_start = time.monotonic()
        self._fps_interval = 5.0

    # -- public factory for tests (inject mock sessions) --------------------
    @classmethod
    def _from_sessions(cls, palm_session, landmark_session):
        """Build a solver with pre-built (or mock) ONNX sessions."""
        obj = object.__new__(cls)
        obj._palm_session = palm_session
        obj._landmark_session = landmark_session
        obj._anchors = ANCHORS
        obj._frame_count = 0
        obj._fps_start = time.monotonic()
        obj._fps_interval = 5.0
        return obj

    # -- FPS logging --------------------------------------------------------
    def _log_fps(self):
        self._frame_count += 1
        elapsed = time.monotonic() - self._fps_start
        if elapsed >= self._fps_interval:
            fps = self._frame_count / elapsed
            logger.info("WebcamPoseSolver: %.1f FPS (%d frames / %.1fs)",
                        fps, self._frame_count, elapsed)
            self._frame_count = 0
            self._fps_start = time.monotonic()

    # -- palm detection -----------------------------------------------------
    def _preprocess_palm(self, gray):
        """Grayscale HxW uint8 -> [1,3,128,128] float32 in [0,1]."""
        resized = _bilinear_resize(gray, PALM_SIZE, PALM_SIZE)
        norm = resized / 255.0
        return np.stack([norm, norm, norm], axis=0)[None]  # [1,3,128,128]

    def _decode_detections(self, raw_regressors, raw_scores):
        """Decode palm-detection outputs after NMS.

        Parameters
        ----------
        raw_regressors : ndarray [1, 896, 18]
        raw_scores     : ndarray [1, 896, 1]

        Returns
        -------
        boxes      : ndarray [N, 4]  (x0,y0,x1,y1 normalised 0-1)
        scores     : ndarray [N]
        keypoints  : ndarray [N, 7, 2]  (x,y normalised 0-1)
        """
        regs = raw_regressors[0]             # [896, 18]
        scores = _sigmoid(raw_scores[0, :, 0])  # [896]

        mask = scores >= DETECTION_THRESHOLD
        if not np.any(mask):
            return (np.empty((0, 4), np.float32),
                    np.empty(0, np.float32),
                    np.empty((0, NUM_PALM_KEYPOINTS, 2), np.float32))

        regs = regs[mask]
        scores = scores[mask]
        anch = self._anchors[mask]

        # decode centres & size (offsets are scaled by PALM_SIZE)
        cx = regs[:, 0] / PALM_SIZE + anch[:, 0]
        cy = regs[:, 1] / PALM_SIZE + anch[:, 1]
        w = regs[:, 2] / PALM_SIZE
        h = regs[:, 3] / PALM_SIZE

        boxes = np.stack([cx - w / 2, cy - h / 2,
                          cx + w / 2, cy + h / 2], axis=-1)

        # decode keypoints
        kps = np.zeros((len(regs), NUM_PALM_KEYPOINTS, 2), np.float32)
        for k in range(NUM_PALM_KEYPOINTS):
            kps[:, k, 0] = regs[:, 4 + k * 2] / PALM_SIZE + anch[:, 0]
            kps[:, k, 1] = regs[:, 4 + k * 2 + 1] / PALM_SIZE + anch[:, 1]

        keep = _nms(boxes, scores)
        return boxes[keep], scores[keep], kps[keep]

    # -- landmark inference -------------------------------------------------
    def _crop_for_landmark(self, gray, box, width, height):
        """Crop a square region around the palm detection and resize to 224x224.

        Parameters
        ----------
        gray   : ndarray [H, W] uint8
        box    : ndarray [4]  (x0,y0,x1,y1 normalised 0-1)
        width  : int -- frame width
        height : int -- frame height

        Returns
        -------
        inp      : ndarray [1, 3, 224, 224] float32
        crop_box : tuple (px0, py0, px1, py1) pixel crop used
        """
        bx0, by0, bx1, by1 = box
        bcx = (bx0 + bx1) / 2
        bcy = (by0 + by1) / 2
        side_norm = max(bx1 - bx0, by1 - by0) * CROP_EXPAND

        # pixel-space square crop
        px0 = int((bcx - side_norm / 2) * width)
        py0 = int((bcy - side_norm / 2) * height)
        px1 = int((bcx + side_norm / 2) * width)
        py1 = int((bcy + side_norm / 2) * height)

        # clamp source reads to image bounds, zero-pad outside
        cpx0 = max(0, px0)
        cpy0 = max(0, py0)
        cpx1 = min(width, px1)
        cpy1 = min(height, py1)
        crop_h = py1 - py0
        crop_w = px1 - px0
        if crop_h < 1 or crop_w < 1:
            # degenerate box -- return blank
            blank = np.zeros((1, 3, LANDMARK_SIZE, LANDMARK_SIZE), np.float32)
            return blank, (px0, py0, px1, py1)

        crop = np.zeros((crop_h, crop_w), dtype=np.uint8)
        crop[cpy0 - py0:cpy1 - py0, cpx0 - px0:cpx1 - px0] = \
            gray[cpy0:cpy1, cpx0:cpx1]

        resized = _bilinear_resize(crop, LANDMARK_SIZE, LANDMARK_SIZE)
        norm = resized / 255.0
        inp = np.stack([norm, norm, norm], axis=0)[None]
        return inp, (px0, py0, px1, py1)

    def _run_landmarks(self, inp, crop_box):
        """Run the landmark model on a cropped hand image.

        Returns
        -------
        landmarks  : ndarray [21, 3]  (x, y in frame px; z relative)
        handedness : float  (0 = left, 1 = right)
        presence   : float  (0-1)
        """
        outs = self._landmark_session.run(
            None, {"input_1": inp})
        raw_lm = outs[0][0].reshape(NUM_LANDMARKS, 3).astype(np.float32)
        handedness = float(_sigmoid(np.array([outs[1][0, 0]]))[0])
        presence = float(_sigmoid(np.array([outs[2][0, 0]]))[0])

        # map from 224x224 crop coords to frame pixels
        px0, py0, px1, py1 = crop_box
        cw = float(px1 - px0)
        ch = float(py1 - py0)
        if cw < 1:
            cw = 1.0
        if ch < 1:
            ch = 1.0
        raw_lm[:, 0] = raw_lm[:, 0] / LANDMARK_SIZE * cw + px0
        raw_lm[:, 1] = raw_lm[:, 1] / LANDMARK_SIZE * ch + py0
        raw_lm[:, 2] = raw_lm[:, 2] / LANDMARK_SIZE * cw  # z scaled by width

        return raw_lm, handedness, presence

    # -- main entry ---------------------------------------------------------
    def __call__(self, gray_frame_bytes: bytes, width: int,
                 height: int) -> List[Hand]:
        """Detect hands in a single grayscale frame.

        Parameters
        ----------
        gray_frame_bytes : bytes
            Raw grayscale pixels, row-major, ``width * height`` bytes.
        width, height : int
            Frame dimensions.

        Returns
        -------
        list[Hand]
            Detected hands, each carrying 21 landmarks in frame-pixel coords.
        """
        self._log_fps()

        # 1. bytes -> numpy
        gray = np.frombuffer(gray_frame_bytes, dtype=np.uint8).reshape(
            height, width)

        # 2-3. palm detection
        palm_inp = self._preprocess_palm(gray)
        regressors, classificators = self._palm_session.run(
            None, {"input": palm_inp})

        # 4. decode + NMS
        boxes, scores, _kps = self._decode_detections(regressors, classificators)
        if len(boxes) == 0:
            return []

        # 5-6. landmark inference per detection
        hands: List[Hand] = []
        for i in range(len(boxes)):
            crop_inp, crop_box = self._crop_for_landmark(
                gray, boxes[i], width, height)
            landmarks, handedness, presence = self._run_landmarks(
                crop_inp, crop_box)

            if presence < LANDMARK_PRESENCE_THRESHOLD:
                continue

            combined = float(scores[i]) * presence
            bbox = (
                float(boxes[i][0] * width),
                float(boxes[i][1] * height),
                float(boxes[i][2] * width),
                float(boxes[i][3] * height),
            )
            hands.append(Hand(
                landmarks=landmarks,
                confidence=combined,
                bbox=bbox,
                handedness=handedness,
            ))

        return hands
