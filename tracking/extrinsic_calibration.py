#!/usr/bin/env python3
"""Extrinsic calibration between the Leap Motion Controller and an external
webcam: estimate the rigid transform (R, t) that maps a 3D point in the Leap's
coordinate frame into the webcam's frame.

Workflow:
  1. Collect (leap_3d, cam_2d) correspondences — the user holds their hand in
     view of both sensors and the pipeline matches the Leap's 3D skeleton to
     the webcam's 2D detections.
  2. Solve with cv2.solvePnP using a geometric prior that encodes the typical
     desk setup (Leap looking up, webcam above looking forward/down).
  3. Persist the result and periodically check for drift.

Coordinates:
  Leap frame : right-handed, mm, +X right, +Y up, -Z away (as emitted by
               the stereo triangulation in calibration.py / skeleton.py).
  Webcam frame: OpenCV camera convention (right-handed, +X right, +Y down,
               +Z forward).
  The solved (R, t) transforms  Leap -> Webcam.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class CalibrationResult:
    """Rigid transform from source (Leap) frame to target (webcam) frame."""
    rotation: np.ndarray       # 3x3 rotation matrix
    translation: np.ndarray    # 3x1 translation vector, mm
    reprojection_error: float  # RMS reprojection error, px

    def transform_point(self, p: np.ndarray) -> np.ndarray:
        """Transform a 3D point from source frame to target frame."""
        p = np.asarray(p, dtype=np.float64).reshape(3)
        return (self.rotation @ p + self.translation.flatten()).reshape(3)

    def inverse(self) -> "CalibrationResult":
        """Return the inverse transform (target -> source)."""
        r_inv = self.rotation.T
        t_inv = -r_inv @ self.translation
        return CalibrationResult(
            rotation=r_inv,
            translation=t_inv.reshape(3, 1),
            reprojection_error=self.reprojection_error,
        )

    def save(self, path: str) -> None:
        """Persist to a .npz file."""
        np.savez(
            path,
            R=self.rotation,
            t=self.translation,
            err=np.array([self.reprojection_error]),
        )

    @classmethod
    def load(cls, path: str) -> "CalibrationResult":
        """Load from a .npz file previously written by save()."""
        d = np.load(path)
        return cls(
            rotation=d["R"],
            translation=d["t"],
            reprojection_error=float(d["err"].flat[0]),
        )


def _initial_rvec_tvec():
    """Geometric prior for a typical desk setup.

    The Leap Motion sits on the desk looking up; a webcam is mounted ~300 mm
    above and ~200 mm behind it, looking forward/down. In this configuration
    the dominant rotation is Rx(-90 deg): Leap +Y (up) maps to webcam -Z
    (behind the camera), and Leap -Z (away) maps to webcam -Y (down in image).
    The translation (0, 300, -200) mm places the webcam origin above and
    behind the Leap.

    This prior gives solvePnP a good basin — on a desk the real transform is
    close, so SOLVEPNP_ITERATIVE converges reliably.
    """
    angle = -np.pi / 2.0
    r_prior = np.array([[1.0, 0.0, 0.0],
                        [0.0, np.cos(angle), -np.sin(angle)],
                        [0.0, np.sin(angle), np.cos(angle)]], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(r_prior)
    tvec = np.array([[0.0], [300.0], [-200.0]], dtype=np.float64)
    return rvec, tvec


class CalibrationManager:
    """Collect 3D-2D correspondences and solve for the Leap-to-webcam extrinsic.

    Parameters
    ----------
    webcam_intrinsics_K : np.ndarray
        3x3 camera matrix of the webcam (from cv2.calibrateCamera or similar).
    dist_coeffs : np.ndarray, optional
        Webcam distortion coefficients. Defaults to zero (pre-undistorted).
    """

    def __init__(self, webcam_intrinsics_K: np.ndarray,
                 dist_coeffs: np.ndarray | None = None):
        self._K = np.asarray(webcam_intrinsics_K, dtype=np.float64).reshape(3, 3)
        self._dist = (np.zeros(5, dtype=np.float64) if dist_coeffs is None
                      else np.asarray(dist_coeffs, dtype=np.float64).ravel())
        self._correspondences_3d: list[np.ndarray] = []
        self._correspondences_2d: list[np.ndarray] = []

    @property
    def num_correspondences(self) -> int:
        return len(self._correspondences_3d)

    def add_correspondence(self, leap_3d: np.ndarray,
                           cam_2d: np.ndarray) -> None:
        """Add one (or many) 3D-to-2D correspondences.

        Parameters
        ----------
        leap_3d : array (3,) or (N, 3)
            3D point(s) in Leap frame, mm.
        cam_2d : array (2,) or (N, 2)
            Corresponding 2D point(s) in webcam image, px.
        """
        pts3 = np.asarray(leap_3d, dtype=np.float64)
        pts2 = np.asarray(cam_2d, dtype=np.float64)
        if pts3.ndim == 1:
            pts3 = pts3.reshape(1, 3)
            pts2 = pts2.reshape(1, 2)
        if pts3.shape[0] != pts2.shape[0]:
            raise ValueError(
                f"3D ({pts3.shape[0]}) and 2D ({pts2.shape[0]}) counts differ")
        for i in range(pts3.shape[0]):
            self._correspondences_3d.append(pts3[i])
            self._correspondences_2d.append(pts2[i])

    def solve(self, min_points: int = 6) -> CalibrationResult:
        """Solve for the rigid transform using all collected correspondences.

        Requires at least ``min_points`` correspondences (6 by default — more
        than the PnP minimum of 4, to reduce noise sensitivity).

        Returns
        -------
        CalibrationResult
            The solved Leap -> Webcam transform.

        Raises
        ------
        ValueError
            If too few correspondences have been added.
        RuntimeError
            If cv2.solvePnP fails to converge.
        """
        n = len(self._correspondences_3d)
        if n < min_points:
            raise ValueError(
                f"Need >= {min_points} correspondences, have {n}")

        obj = np.array(self._correspondences_3d, dtype=np.float64)
        img = np.array(self._correspondences_2d, dtype=np.float64)

        rvec0, tvec0 = _initial_rvec_tvec()

        ok, rvec, tvec = cv2.solvePnP(
            obj, img, self._K, self._dist,
            rvec=rvec0.copy(), tvec=tvec0.copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise RuntimeError("cv2.solvePnP failed to converge")

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)

        # compute RMS reprojection error
        projected, _ = cv2.projectPoints(obj, rvec, tvec, self._K, self._dist)
        projected = projected.reshape(-1, 2)
        err = np.sqrt(np.mean(np.sum((projected - img) ** 2, axis=1)))

        return CalibrationResult(rotation=R, translation=t,
                                 reprojection_error=float(err))

    def check_drift(self, leap_3d: np.ndarray, cam_2d: np.ndarray,
                    result: CalibrationResult,
                    threshold_px: float = 15.0) -> bool:
        """Check whether the calibration has drifted.

        Projects the given 3D point(s) through the solved calibration and
        compares to the observed 2D location(s). Returns True if the mean
        reprojection error exceeds ``threshold_px``.
        """
        pts3 = np.asarray(leap_3d, dtype=np.float64).reshape(-1, 3)
        pts2 = np.asarray(cam_2d, dtype=np.float64).reshape(-1, 2)

        rvec, _ = cv2.Rodrigues(result.rotation)
        projected, _ = cv2.projectPoints(
            pts3, rvec, result.translation, self._K, self._dist)
        projected = projected.reshape(-1, 2)
        mean_err = float(np.mean(np.linalg.norm(projected - pts2, axis=1)))
        return mean_err > threshold_px
