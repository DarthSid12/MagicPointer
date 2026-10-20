"""Metric stereo reconstruction and startup wall-marker verification."""
import json
from pathlib import Path

import cv2
import numpy as np


def markers(image):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
    return {} if ids is None else {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


class Stereo:
    def __init__(self, data):
        self.data = data
        if data.get("version") != 1:
            raise ValueError("Unsupported calibration file version")
        try:
            self.size = tuple(data["image_size"])
            self.k1, self.k2 = (np.asarray(data[key], dtype=float).reshape(3, 3) for key in ("k1", "k2"))
            self.d1, self.d2 = (np.asarray(data[key], dtype=float).reshape(-1) for key in ("d1", "d2"))
            self.r = np.asarray(data["rotation"], dtype=float).reshape(3, 3)
            self.t = np.asarray(data["translation"], dtype=float).reshape(3, 1)
            arrays = (self.k1, self.k2, self.d1, self.d2, self.r, self.t)
            if not all(np.isfinite(a).all() for a in arrays):
                raise ValueError("Calibration contains nonfinite numbers")
            if len(self.size) != 2 or any(not isinstance(v, int) or v <= 0 for v in self.size):
                raise ValueError("Invalid calibrated image size")
            if any(len(d) not in (4, 5, 8, 12, 14) for d in (self.d1, self.d2)):
                raise ValueError("Invalid lens distortion coefficients")
            if not np.allclose(self.r.T @ self.r, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(self.r), 1., atol=1e-4):
                raise ValueError("Invalid stereo rotation")
            if not .01 <= np.linalg.norm(self.t) <= 3:
                raise ValueError("Camera baseline must be between 1 cm and 3 m")
            for k in (self.k1, self.k2):
                if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
                    raise ValueError("Invalid camera intrinsics")
            rms = np.asarray(data["rms"], dtype=float)
            if rms.shape != (3,) or not np.isfinite(rms).all() or (rms < 0).any() or (rms > 2.).any():
                raise ValueError("Calibration RMS exceeds 2 pixels; capture better checkerboard views")
            references = data.get("wall_markers")
            if references is not None:
                if len(references) != 2:
                    raise ValueError("Calibration must reference wall markers in both cameras")
                for reference in references:
                    for marker_id in ('0', '1'):
                        points = np.asarray(reference[marker_id], dtype=float)
                        if points.shape != (4, 2) or not np.isfinite(points).all():
                            raise ValueError("Invalid calibrated wall-marker corners")
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("Incomplete calibration file; run calibration again") from error
        self.p1 = np.column_stack((np.eye(3), np.zeros(3)))
        self.p2 = np.column_stack((self.r, self.t))

    @classmethod
    def load(cls, path):
        try:
            return cls(json.loads(Path(path).read_text()))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot load calibration {path}; run the calibrate command first") from error

    def reconstruct(self, left, right, max_error=4.):
        a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        if a.ndim != 2 or a.shape[1] != 2 or a.shape != b.shape or not len(a):
            return None
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            return None
        u = cv2.undistortPoints(a.reshape(-1, 1, 2), self.k1, self.d1).reshape(-1, 2)
        v = cv2.undistortPoints(b.reshape(-1, 1, 2), self.k2, self.d2).reshape(-1, 2)
        homogeneous = cv2.triangulatePoints(self.p1, self.p2, u.T, v.T)
        if (np.abs(homogeneous[3]) < 1e-9).any():
            return None
        points = (homogeneous[:3] / homogeneous[3]).T
        other = (self.r @ points.T + self.t).T
        if not np.isfinite(points).all() or (points[:, 2] <= .05).any() or (other[:, 2] <= .05).any():
            return None
        if (np.linalg.norm(points, axis=1) > 6.).any():
            return None
        for observed, k, d, r, t in (
            (a, self.k1, self.d1, np.zeros(3), np.zeros(3)),
            (b, self.k2, self.d2, cv2.Rodrigues(self.r)[0], self.t),
        ):
            projected = cv2.projectPoints(points, r, t, k, d)[0].reshape(-1, 2)
            if (np.linalg.norm(projected - observed, axis=1) > max_error).any():
                return None
        return points

    def check_frames(self, frames):
        if len(frames) != 2 or any(tuple(f.image.shape[1::-1]) != self.size for f in frames):
            raise ValueError("Camera resolution differs from calibration; recalibrate at this resolution")
        return abs(frames[0].grabbed_at - frames[1].grabbed_at) <= .04

    def check_markers(self, images, require_visible=True):
        references = self.data.get("wall_markers")
        if not references:
            raise ValueError("Calibration has no wall-marker reference; run calibration again")
        detected = [markers(image) for image in images]
        for camera, (found, reference) in enumerate(zip(detected, references)):
            for marker_id in (0, 1):
                if marker_id not in found:
                    if require_visible:
                        return False, f"Show wall markers 0 and 1 to camera {camera}"
                    continue
                expected = np.asarray(reference[str(marker_id)], dtype=float)
                if np.max(np.linalg.norm(found[marker_id] - expected, axis=1)) > 8.:
                    return False, "Rig or wall markers moved; recalibrate before control"
        return True, "Wall markers verified"


def calibrate_views(object_points, left, right, size):
    if len(object_points) < 15:
        raise ValueError("Capture at least 15 varied checkerboard views")
    rms1, k1, d1, _, _ = cv2.calibrateCamera(object_points, left, size, None, None)
    rms2, k2, d2, _, _ = cv2.calibrateCamera(object_points, right, size, None, None)
    rms, _, _, _, _, rotation, translation, _, _ = cv2.stereoCalibrate(
        object_points, left, right, k1, d1, k2, d2, size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    data = {"version": 1, "image_size": list(size), "k1": k1.tolist(), "k2": k2.tolist(),
            "d1": d1.ravel().tolist(), "d2": d2.ravel().tolist(), "rotation": rotation.tolist(),
            "translation": translation.ravel().tolist(), "rms": [rms1, rms2, rms]}
    return Stereo(data)
