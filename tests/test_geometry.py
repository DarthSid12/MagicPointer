import copy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from magicpointer.cameras import Frame
from magicpointer.stereo import Stereo, calibrate_views, markers


def rig():
    k = [[1200., 0., 960.], [0., 1200., 540.], [0., 0., 1.]]
    return {"version": 1, "image_size": [1920, 1080], "k1": k, "k2": k,
            "d1": [0.]*5, "d2": [0.]*5, "rotation": np.eye(3).tolist(),
            "translation": [-.3, 0., 0.], "rms": [.1, .1, .2]}


def project(stereo, points):
    left = cv2.projectPoints(points, np.zeros(3), np.zeros(3), stereo.k1, stereo.d1)[0].reshape(-1, 2)
    right = cv2.projectPoints(points, cv2.Rodrigues(stereo.r)[0], stereo.t, stereo.k2, stereo.d2)[0].reshape(-1, 2)
    return left, right


class GeometryTests(unittest.TestCase):
    def test_reconstruct_across_distances_and_room_positions(self):
        stereo = Stereo(rig())
        points = np.array([[x, y, z] for x in (-.5, 0., .5) for y in (-.3, .3) for z in (.7, 2., 3.6576)])
        a, b = project(stereo, points)
        np.testing.assert_allclose(stereo.reconstruct(a, b), points, atol=1e-9)

    def test_rotated_distorted_rig(self):
        data = rig()
        data['rotation'] = cv2.Rodrigues(np.array([.03, -.10, .01]))[0].tolist()
        data['d1'] = [-.12, .02, .001, -.002, 0.]
        data['d2'] = [-.1, .01, -.001, .002, 0.]
        stereo = Stereo(data)
        points = np.array([[.1, .2, 2.], [-.3, -.1, 3.]])
        np.testing.assert_allclose(stereo.reconstruct(*project(stereo, points)), points, atol=1e-5)

    def test_outliers_zero_disparity_and_behind_cameras_rejected(self):
        stereo = Stereo(rig())
        a, b = project(stereo, np.array([[.1, .2, 2.]]))
        bad = b.copy()
        bad[0, 1] += 30
        self.assertIsNone(stereo.reconstruct(a, bad))
        self.assertIsNone(stereo.reconstruct(a, a))
        self.assertIsNone(stereo.reconstruct(b, a))
        self.assertIsNone(stereo.reconstruct([[float('nan'), 0.]], [[0., 0.]]))

    def test_calibration_rejects_invalid_geometry(self):
        for key, value in [('rms', [0., 0., 20.]), ('translation', [0., 0., 0.]),
                           ('rotation', np.zeros((3, 3)).tolist()), ('k1', [[0.]*3]*3)]:
            data = copy.deepcopy(rig())
            data[key] = value
            with self.assertRaises(ValueError):
                Stereo(data)

    def test_resolution_and_pair_timing(self):
        stereo = Stereo(rig())
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.assertTrue(stereo.check_frames([Frame(image, 1.), Frame(image, 1.02)]))
        self.assertFalse(stereo.check_frames([Frame(image, 1.), Frame(image, 1.1)]))
        with self.assertRaises(ValueError):
            stereo.check_frames([Frame(image[:500], 1.), Frame(image, 1.)])

    def test_solve_calibration_from_synthetic_varied_views(self):
        stereo = Stereo(rig())
        grid = np.zeros((54, 3), dtype=np.float32)
        grid[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * .025
        objects, left, right = [], [], []
        rng = np.random.default_rng(42)
        for _ in range(18):
            rotation = rng.uniform(-.4, .4, 3)
            translation = np.array([rng.uniform(-.3, .2), rng.uniform(-.2, .1), rng.uniform(.8, 1.5)])
            points = (cv2.Rodrigues(rotation)[0] @ grid.T).T + translation
            a, b = project(stereo, points)
            objects.append(grid.copy())
            left.append(a.astype(np.float32).reshape(-1, 1, 2))
            right.append(b.astype(np.float32).reshape(-1, 1, 2))
        solved = calibrate_views(objects, left, right, (1920, 1080))
        np.testing.assert_allclose(solved.t, stereo.t, atol=.001)
        self.assertLess(solved.data['rms'][2], .01)

    def test_actual_aruco_detection_and_startup_movement_check(self):
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        image = np.full((500, 900, 3), 255, np.uint8)
        for i, x in enumerate((100, 550)):
            marker = cv2.aruco.generateImageMarker(dictionary, i, 180)
            image[100:280, x:x+180] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        detected = markers(image)
        self.assertEqual(set(detected), {0, 1})
        data = rig()
        data['wall_markers'] = [{str(i): detected[i].tolist() for i in (0, 1)}]*2
        stereo = Stereo(data)
        self.assertTrue(stereo.check_markers([image, image])[0])
        shifted = np.roll(image, 30, axis=1)
        self.assertFalse(stereo.check_markers([shifted, image])[0])
        blank = np.full_like(image, 255)
        self.assertFalse(stereo.check_markers([blank, image])[0])
        self.assertTrue(stereo.check_markers([blank, image], require_visible=False)[0])

    def test_file_missing_and_corrupt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rig.json'
            with self.assertRaises(ValueError):
                Stereo.load(path)
            path.write_text('{bad')
            with self.assertRaises(ValueError):
                Stereo.load(path)
