import threading
import time
import unittest
from unittest.mock import patch

import numpy as np

from magicpointer import cameras
from magicpointer.cameras import Cameras


class FakeCapture:
    """Free-running device: read() blocks for one frame period like real drivers."""

    def __init__(self, period=.01, failing=None):
        self.period, self.failing = period, failing or (lambda count: False)
        self.count, self.released, self.lock = 0, 0, threading.Lock()

    def isOpened(self):
        return True

    def set(self, *_):
        return True

    def get(self, _):
        return 30.

    def read(self):
        time.sleep(self.period)
        with self.lock:
            self.count += 1
            count = self.count
        if self.failing(count):
            return False, None
        return True, np.full((4, 4, 3), count % 256, np.uint8)

    def release(self):
        self.released += 1


def open_with(*devices):
    return patch('cv2.VideoCapture', side_effect=list(devices))


class StreamTests(unittest.TestCase):
    def test_reads_newest_frame_without_waiting_or_repeating(self):
        device = FakeCapture(period=.005)
        with open_with(device), Cameras(0, fps='30', width=4, height=4) as camera:
            first = camera.read()[0]
            time.sleep(.08)
            second = camera.read()[0]
            # Frames produced while the caller was busy are skipped, not queued.
            self.assertGreater(second.sequence - first.sequence, 3)
            self.assertGreater(second.grabbed_at, time.monotonic() - .03)
            sequences = [camera.read()[0].sequence for _ in range(5)]
            self.assertEqual(sequences, sorted(set(sequences)))
            self.assertGreater(sequences[0], second.sequence)
        self.assertEqual(device.released, 1)

    def test_lagging_camera_gets_one_frame_to_tighten_the_pair(self):
        fast, slow = FakeCapture(period=.005), FakeCapture(period=.05)
        with open_with(fast, slow), Cameras(0, 1, fps='30', width=4, height=4) as camera:
            skews = []
            for _ in range(10):
                left, right = camera.read()
                skews.append(abs(left.grabbed_at - right.grabbed_at))
            self.assertLessEqual(np.median(skews), Cameras.pair_tolerance)

    def test_brief_failure_reopens_device_and_keeps_streaming(self):
        broken = FakeCapture(failing=lambda count: count > 3)
        replacement = FakeCapture()
        with patch.object(cameras._Stream, 'reopen_after', .05), open_with(broken, replacement), \
                Cameras(0, fps='30', width=4, height=4) as camera:
            before = camera.read()[0].sequence
            deadline = time.monotonic() + 2.
            while replacement.count < 5 and time.monotonic() < deadline:
                frame = camera.read()[0]
            self.assertGreater(frame.sequence, before)
            self.assertGreaterEqual(replacement.count, 5)
            self.assertEqual(broken.released, 1)
        self.assertEqual(replacement.released, 1)

    def test_persistent_failure_raises_readable_error(self):
        dead = FakeCapture(failing=lambda count: count > 1)
        with patch.object(cameras._Stream, 'reopen_after', 10.), patch.object(cameras._Stream, 'give_up', .2), \
                open_with(dead), Cameras(0, fps='30', width=4, height=4) as camera:
            camera.read()
            started = time.monotonic()
            with self.assertRaisesRegex(ValueError, 'Camera stream stopped'):
                camera.read()
            self.assertLess(time.monotonic() - started, 2.)


if __name__ == '__main__':
    unittest.main()
