import unittest

from magicpointer.desktop import SmoothDesktop


class FakeNative:
    wheel_step = 1 / 120

    def __init__(self):
        self.moves, self.clicks, self.wheel = [], [], []

    def pixel(self, point):
        return round(point[0] * 1919), round(point[1] * 1079)

    def move(self, point):
        self.moves.append(self.pixel(point))

    def click(self, button):
        self.clicks.append((button, self.moves[-1]))

    def scroll(self, lines):
        self.wheel.append(lines)

    def scroll_fine(self, lines):
        self.wheel.append(lines)


class SmoothDesktopTests(unittest.TestCase):
    def setUp(self):
        self.now = [0.]
        self.native = FakeNative()
        self.desktop = SmoothDesktop(self.native, clock=lambda: self.now[0], start=False)

    def frame(self, point, frame_seconds=1 / 15, ticks=8):
        self.now[0] += frame_seconds
        self.desktop.move(point)
        for _ in range(ticks):
            self.desktop.tick(frame_seconds / ticks)

    def test_pointer_glides_in_small_steps_between_vision_frames(self):
        self.frame((.5, .5))
        before = len(self.native.moves)
        self.frame((.6, .5))
        steps = self.native.moves[before:]
        self.assertGreater(len(steps), 4)
        self.assertTrue(all(b[0] >= a[0] for a, b in zip(steps, steps[1:])))
        self.assertLess(max(b[0] - a[0] for a, b in zip(steps, steps[1:])), .1 * 1919 / 2)
        # Converges on the target well before the next frames arrive.
        for _ in range(3):
            self.frame((.6, .5))
        self.assertEqual(self.native.moves[-1], self.native.pixel((.6, .5)))

    def test_click_lands_exactly_on_latest_target(self):
        self.frame((.2, .2))
        self.desktop.move((.8, .3))  # anchor arrives with the click, before any glide tick
        self.desktop.click('left')
        self.assertEqual(self.native.clicks, [('left', self.native.pixel((.8, .3)))])

    def test_no_os_calls_once_pointer_has_settled(self):
        for _ in range(4):
            self.frame((.4, .4))
        count = len(self.native.moves)
        for _ in range(50):
            self.desktop.tick(1 / 120)
        self.assertEqual(len(self.native.moves), count)

    def test_scroll_lines_spread_into_fine_wheel_steps_without_losing_amount(self):
        self.desktop.scroll(3)
        for _ in range(120):
            self.desktop.tick(1 / 120)
        self.assertGreater(len(self.native.wheel), 10)
        self.assertAlmostEqual(sum(self.native.wheel), 3., places=6)
        self.assertTrue(all(0 < amount < 1 for amount in self.native.wheel))

    def test_line_scroll_mode_passes_whole_lines_through(self):
        desktop = SmoothDesktop(self.native, smooth_scroll=False, start=False)
        desktop.scroll(-2)
        self.assertEqual(self.native.wheel, [-2])

    def test_background_os_error_is_reported_on_next_input_call(self):
        self.desktop.error = OSError('Windows rejected pointer input')
        with self.assertRaisesRegex(OSError, 'rejected'):
            self.desktop.move((.5, .5))
        self.desktop.move((.5, .5))

    def test_thread_starts_and_stops(self):
        desktop = SmoothDesktop(self.native)
        desktop.move((.3, .3))
        desktop.close()
        self.assertFalse(desktop.thread.is_alive())


if __name__ == '__main__':
    unittest.main()
