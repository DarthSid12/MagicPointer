"""Model landmarks and rotation-invariant joint geometry classification."""
from dataclasses import dataclass
import math

from .engine import Gesture


def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def angle(a, b, c):
    ab = [x - y for x, y in zip(a, b)]
    cb = [x - y for x, y in zip(c, b)]
    denominator = math.sqrt(sum(v*v for v in ab) * sum(v*v for v in cb))
    if denominator < 1e-10:
        return 0.0
    return math.degrees(math.acos(max(-1., min(1., sum(x*y for x, y in zip(ab, cb)) / denominator))))


def gesture_metrics(points):
    """Expose the exact current classifier measurements for labelled recordings."""
    if len(points) != 21 or not all(math.isfinite(v) for p in points for v in p):
        return {"valid": False, "reason": "invalid landmarks", "gesture": Gesture.UNKNOWN.value}
    palm = distance(points[0], points[9])
    if palm < 1e-6:
        return {"valid": False, "reason": "degenerate palm", "gesture": Gesture.UNKNOWN.value}
    fingers = []
    measurements = {}
    for name, mcp in zip(("index", "middle", "ring", "little"), (5, 9, 13, 17)):
        pip = angle(points[mcp], points[mcp+1], points[mcp+2])
        dip = angle(points[mcp+1], points[mcp+2], points[mcp+3])
        reach = distance(points[mcp+3], points[0]) / max(distance(points[mcp], points[0]), 1e-12)
        curl = distance(points[mcp+3], points[0]) / max(distance(points[mcp+1], points[0]), 1e-12)
        straight = pip > 155 and dip > 150
        extended = straight and distance(points[mcp+3], points[0]) > distance(points[mcp], points[0]) * 1.25
        curled = distance(points[mcp+3], points[0]) < distance(points[mcp+1], points[0]) * 1.05
        fingers.append(1 if extended else (0 if curled else -1))
        measurements[name] = {"pip_degrees": pip, "dip_degrees": dip, "reach_ratio": reach,
                              "curl_ratio": curl, "state": fingers[-1]}
    thumb_angle = angle(points[2], points[3], points[4])
    thumb_straight = thumb_angle > 150
    separation = distance(points[4], points[5]) / palm
    thumb = 1 if thumb_straight and separation > .65 else (0 if separation < .55 else -1)
    signature = (thumb, *fingers)
    gesture = {
        (0, 0, 0, 0, 0): Gesture.FIST,
        (0, 1, 0, 0, 0): Gesture.POINT,
        (1, 1, 0, 0, 0): Gesture.LEFT,
        (1, 1, 1, 0, 0): Gesture.RIGHT,
        (0, 1, 1, 0, 0): Gesture.SCROLL,
        (1, 1, 1, 1, 1): Gesture.PALM,
    }.get(signature, Gesture.UNKNOWN)
    return {"valid": True, "palm_size": palm, "fingers": measurements,
            "thumb": {"ip_degrees": thumb_angle, "separation_ratio": separation, "state": thumb},
            "signature": list(signature), "gesture": gesture.value}


def classify(points):
    return Gesture(gesture_metrics(points)["gesture"])


@dataclass
class Hand:
    pixels: object
    world: object
    label: str
    gesture: Gesture
    handedness_score: float = None
    image_landmarks: object = None


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
        category = result.handedness[0][0]
        normalized = [(p.x, p.y, p.z) for p in result.hand_landmarks[0]]
        return Hand(pixels, world, category.category_name, classify(world), category.score, normalized)

    def close(self):
        self.detector.close()
