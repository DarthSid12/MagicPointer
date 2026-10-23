from concurrent.futures import ThreadPoolExecutor
import threading
import time
import unittest
from unittest.mock import MagicMock

import numpy as np

from magicpointer.app import MarkerWatch, detect_views
from magicpointer.cameras import Frame
from magicpointer.ownership import HandOwner
from magicpointer.recovery import CropRecovery
from test_ownership_fusion import pose


def frames(count=2, size=(1080, 1920)):
    return [Frame(np.zeros((*size, 3), np.uint8), 1. + i * .005) for i in range(count)]


class CropTracker:
    """Finds `hand` (given in full-frame pixels) only if it lies inside the crop it is shown."""

    def __init__(self, hand, origin):
        self.hand, self.origin, self.crops = hand, origin, []

    def detect_all(self, image, timestamp):
        left, top = self.origin()
        self.crops.append((left, top, image.shape[1], image.shape[0]))
        pixels = np.asarray(self.hand.pixels) - [left, top]
        if (pixels < 0).any() or (pixels[:, 0] >= image.shape[1]).any() or (pixels[:, 1] >= image.shape[0]).any():
            return []
        found = pose(self.hand.gesture, label=self.hand.label)
        found.pixels = pixels
        found.image_landmarks = [(x / image.shape[1], y / image.shape[0], 0.) for x, y in pixels]
        return [found]

    def close(self):
        pass


class CropRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.hand = pose(x=1500.)
        self.recovery = CropRecovery(lambda: self.tracker, 2)
        self.origins = []
        original = self.recovery.crop

        def crop(image, box):
            result = original(image, box)
            self.origins.append(result[1:])
            return result
        self.recovery.crop = crop
        self.tracker = CropTracker(self.hand, lambda: self.origins[-1])

    def test_lost_hand_is_found_in_a_magnified_crop_and_mapped_back(self):
        self.recovery.remember(1, self.hand, 1.)
        result = self.recovery.apply([[pose()], []], frames(), 1.1)
        self.assertEqual(len(result[1]), 1)
        np.testing.assert_allclose(result[1][0].pixels, self.hand.pixels)
        left, top, width, height = self.tracker.crops[-1]
        self.assertEqual(width, height)
        self.assertLess(width, 1080)  # magnified: far smaller than the full frame
        x, y, _ = result[1][0].image_landmarks[0]
        self.assertAlmostEqual(x * 1920, self.hand.pixels[0][0])
        self.assertAlmostEqual(y * 1080, self.hand.pixels[0][1])

    def test_no_search_while_detected_without_memory_or_after_expiry(self):
        self.recovery.apply([[pose()], []], frames(), 1.)
        self.recovery.remember(1, self.hand, 1.)
        self.recovery.apply([[pose()], [self.hand]], frames(), 1.1)
        self.recovery.apply([[pose()], []], frames(), 1. + CropRecovery.memory + .1)
        self.assertEqual(self.tracker.crops, [])

    def test_hand_far_from_the_lost_position_is_not_adopted(self):
        self.recovery.remember(1, self.hand, 1.)
        self.tracker.hand = pose(x=1500.)
        self.tracker.hand.pixels = np.asarray(self.tracker.hand.pixels) + [0., 200.]
        # Shift the found hand well outside the remembered box but inside the crop.
        recovery = self.recovery
        recovery.scale = 6.
        result = recovery.apply([[pose()], []], frames(), 1.1)
        self.assertEqual(result[1], [])

    def test_hand_already_detected_nearby_is_not_duplicated(self):
        self.recovery.remember(1, self.hand, 1.)
        moved = pose(x=1500.)
        # The regular detector found the hand just outside the "already seen" zone.
        moved.pixels = np.asarray(self.hand.pixels) + [0., 75.]
        self.tracker.hand = pose(x=1500.)
        self.tracker.hand.pixels = moved.pixels + [3., 3.]
        result = self.recovery.apply([[pose()], [moved]], frames(), 1.1)
        self.assertEqual(len(self.tracker.crops), 1)
        self.assertEqual(result[1], [moved])

    def test_crop_is_clamped_inside_the_frame(self):
        image = np.zeros((480, 640, 3), np.uint8)
        crop, left, top = self.recovery.crop(image, (600., 450., 640., 480.))
        self.assertEqual(crop.shape[:2], (256, 256))
        self.assertEqual((left, top), (384, 224))

    def test_recovered_hand_keeps_its_ownership_track(self):
        owner = HandOwner(2)
        first = owner.select([[pose(x=400.)], [self.hand]], frames(), 1.)
        owner.lock(first.key)
        self.recovery.remember(1, first.hands[1], 1.)
        detections = self.recovery.apply([[pose(x=400.)], []], frames(), 1.03)
        again = owner.select(detections, frames(), 1.03)
        self.assertEqual(again.key, first.key)
        self.assertIsNotNone(again.combined)


class MarkerWatchTests(unittest.TestCase):
    def test_single_bad_check_is_tolerated_and_repeated_ones_stop(self):
        stereo = MagicMock()
        results = iter([(True, ''), (False, 'Rig or wall markers moved'), (True, ''),
                        (False, 'Rig or wall markers moved')] + [(False, 'Rig or wall markers moved')] * 3)
        stereo.check_markers.side_effect = lambda *a, **k: next(results)
        watch = MarkerWatch(stereo, interval=0.)
        now = 0.
        with self.assertRaisesRegex(ValueError, 'moved'):
            for _ in range(40):
                now += .01
                watch.poll([None, None], now)
                if watch.pending:
                    watch.pending.result()
        # Stops on the third consecutive failure (sixth check); one isolated failure was ignored.
        self.assertEqual(stereo.check_markers.call_count, 6)
        watch.close()

    def test_slow_check_does_not_block_the_loop(self):
        stereo = MagicMock()
        release = threading.Event()
        stereo.check_markers.side_effect = lambda *a, **k: (release.wait(2.), '')
        watch = MarkerWatch(stereo)
        started = time.perf_counter()
        watch.poll([None, None], 0.)
        watch.poll([None, None], 5.)
        self.assertLess(time.perf_counter() - started, .1)
        release.set()
        watch.close()


class ParallelDetectionTests(unittest.TestCase):
    def test_parallel_views_return_in_camera_order_with_timings(self):
        trackers = [MagicMock(), MagicMock()]
        trackers[0].detect_all.side_effect = lambda *a: (time.sleep(.05), ['left'])[1]
        trackers[1].detect_all.side_effect = lambda *a: (time.sleep(.05), ['right'])[1]
        with ThreadPoolExecutor(max_workers=2) as pool:
            started = time.perf_counter()
            hands, timings = detect_views(trackers, frames(), pool)
            elapsed = time.perf_counter() - started
        self.assertEqual(hands, [['left'], ['right']])
        self.assertEqual(len(timings), 2)
        self.assertLess(elapsed, .09)


if __name__ == '__main__':
    unittest.main()
