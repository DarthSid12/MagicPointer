"""Model landmarks and rotation-invariant hand-shape classification.

Each digit is reduced to one scale-free extension ratio measured over long spans
of the hand. Long spans tolerate per-landmark jitter far better than joint angles
at short phalanges. Hysteresis keeps a borderline digit in its previous state
instead of rejecting the whole pose.
"""
from dataclasses import dataclass
import math

from .engine import Gesture

# (curl below, extend above) per digit: thumb, index, middle, ring, little.
# Thumb: tip-to-middle-knuckle distance / palm length (wrist to middle knuckle).
# Fingers: tip-to-wrist distance / knuckle-to-wrist distance.
# Initial values from hand geometry; tune against recorded real-hand sessions.
THRESHOLDS = ((.70, .85), (1.4, 1.6), (1.4, 1.6), (1.4, 1.6), (1.3, 1.5))

SHAPES = {
    (0, 0, 0, 0, 0): Gesture.FIST,
    (0, 1, 0, 0, 0): Gesture.POINT,
    (1, 1, 0, 0, 0): Gesture.LEFT,
    (1, 1, 1, 0, 0): Gesture.RIGHT,
    (0, 1, 1, 0, 0): Gesture.SCROLL,
    (1, 1, 1, 1, 1): Gesture.PALM,
}


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def extension(points):
    """Return five extension ratios (thumb first), or None for unusable landmarks."""
    if len(points) != 21 or not all(math.isfinite(v) for p in points for v in p):
        return None
    palm = distance(points[0], points[9])
    if palm < 1e-6:
        return None
    ratios = [distance(points[4], points[9]) / palm]
    for mcp in (5, 9, 13, 17):
        base = distance(points[mcp], points[0])
        if base < 1e-6:
            return None
        ratios.append(distance(points[mcp+3], points[0]) / base)
    return tuple(ratios)


def classify(points):
    """Stateless single-frame classification at the middle of each threshold band."""
    ratios = extension(points)
    if ratios is None:
        return Gesture.UNKNOWN
    return SHAPES.get(tuple(int(r >= (lo + hi) / 2) for r, (lo, hi) in zip(ratios, THRESHOLDS)),
                      Gesture.UNKNOWN)


class Classifier:
    """Per-hand temporal smoothing and per-digit hysteresis.

    Ratios are low-passed over time, then each digit only changes state once it
    clearly crosses the far side of its band. Stereo callers fuse both cameras'
    ratios before calling update so one noisy view cannot veto the other.
    """

    def __init__(self, smoothing=.05, reset_after=.25):
        if not all(math.isfinite(v) and v > 0 for v in (smoothing, reset_after)):
            raise ValueError("Classifier timings must be positive and finite")
        self.smoothing, self.reset_after = smoothing, reset_after
        self._ratios = self._states = self._time = None

    def update(self, ratios, now):
        if ratios is None:
            return Gesture.UNKNOWN
        if self._time is None or not 0 <= now - self._time <= self.reset_after:
            # A fresh or reacquired hand starts from the band midpoints.
            self._ratios = tuple(ratios)
            self._states = tuple(int(r >= (lo + hi) / 2) for r, (lo, hi) in zip(ratios, THRESHOLDS))
        else:
            alpha = 1 - math.exp(-(now - self._time) / self.smoothing)
            self._ratios = tuple(a + alpha * (b - a) for a, b in zip(self._ratios, ratios))
            self._states = tuple(1 if r > hi else 0 if r < lo else state
                                 for r, (lo, hi), state in zip(self._ratios, THRESHOLDS, self._states))
        self._time = now
        return SHAPES.get(self._states, Gesture.UNKNOWN)


@dataclass
class Hand:
    pixels: object
    world: object
    label: str
    gesture: Gesture
    ratios: object = None


class Tracker:
    def __init__(self, model):
        import mediapipe as mp
        from .model import verify
        self.mp = mp
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(verify(model)),
                                                  delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=1, min_hand_detection_confidence=.65,
                min_hand_presence_confidence=.65, min_tracking_confidence=.65))
        self.timestamp = -1

    def detect(self, image, timestamp):
        import cv2
        import numpy as np
        self.timestamp = max(self.timestamp + 1, int(timestamp * 1000))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.detector.detect_for_video(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb), self.timestamp)
        if not result.hand_landmarks:
            return None
        height, width = image.shape[:2]
        pixels = np.array([(p.x * width, p.y * height) for p in result.hand_landmarks[0]])
        world = [(p.x, p.y, p.z) for p in result.hand_world_landmarks[0]]
        return Hand(pixels, world, result.handedness[0][0].category_name, classify(world), extension(world))

    def close(self):
        self.detector.close()
