"""Model landmarks and rotation-invariant joint geometry classification."""
from dataclasses import dataclass
import math

from .engine import Gesture

CLASSIFIER_VERSION = 2
# Extra thumb IP straightening (degrees) required before a tucked thumb counts as open.
THUMB_HYSTERESIS_DEGREES = 5.

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
        chain = sum(distance(points[i], points[i+1]) for i in range(mcp, mcp+3))
        straightness = distance(points[mcp], points[mcp+3]) / max(chain, 1e-12)
        # Real extended fingers retain a natural bend and model DIP estimates
        # vary with view. Require reach and an unfolded chain, not two near-180° joints.
        extended = reach > 1.25 and curl > 1.10 and straightness > .80
        curled = distance(points[mcp+3], points[0]) < distance(points[mcp+1], points[0]) * 1.05
        fingers.append(1 if extended else (0 if curled else -1))
        measurements[name] = {"pip_degrees": pip, "dip_degrees": dip, "reach_ratio": reach,
                              "curl_ratio": curl, "straightness": straightness, "state": fingers[-1]}
    thumb_angle = angle(points[2], points[3], points[4])
    separation = distance(points[4], points[5]) / palm
    # Thumb/index spacing overlaps strongly between open and tucked poses.
    # IP bend separates the supplied examples; a small spacing guard also keeps
    # a straight thumb laid across the palm from being counted as extended.
    thumb = 1 if thumb_angle > 145 and separation > .30 else (0 if thumb_angle < 140 or separation < .30 else -1)
    signature = (thumb, *fingers)
    gesture = SHAPES.get(signature, Gesture.UNKNOWN)
    return {"valid": True, "classifier_version": CLASSIFIER_VERSION, "palm_size": palm, "fingers": measurements,
            "thumb": {"ip_degrees": thumb_angle, "separation_ratio": separation, "state": thumb},
            "signature": list(signature), "gesture": gesture.value}


def classify(points):
    return Gesture(gesture_metrics(points)["gesture"])


class PoseStabilizer:
    """Per-hand hysteresis over the classifier's own measurements.

    Fresh hands use the classifier unchanged. Afterwards an uncertain digit keeps
    its last known state, and a tucked thumb needs THUMB_HYSTERESIS_DEGREES past
    the open threshold to open. Closing uses the classifier's own threshold, so a
    spurious open estimate is never sticky enough to hold a click pose. Clear
    changes still switch on the first frame.
    """

    def __init__(self, reset_after=.25):
        self.reset_after = reset_after
        self.signature = self.time = None

    def update(self, points, now):
        metrics = gesture_metrics(points)
        if not metrics["valid"]:
            return Gesture.UNKNOWN
        previous = self.signature
        if previous is not None and not 0 <= now - self.time <= self.reset_after:
            previous = None
        signature = list(metrics["signature"])
        if previous is not None:
            if previous[0] == 0 and signature[0] == 1 and metrics["thumb"]["ip_degrees"] <= 145 + THUMB_HYSTERESIS_DEGREES:
                signature[0] = 0
            signature = [old if new == -1 else new for new, old in zip(signature, previous)]
        self.signature, self.time = tuple(signature), now
        return SHAPES.get(self.signature, Gesture.UNKNOWN)


@dataclass
class Hand:
    pixels: object
    world: object
    label: str
    gesture: Gesture
    handedness_score: float = None
    image_landmarks: object = None


class Tracker:
    def __init__(self, model, num_hands=1, *, detection_confidence=.65, presence_confidence=.65, image_mode=False):
        import mediapipe as mp
        from .model import verify
        self.mp = mp
        self.image_mode = image_mode
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(verify(model)),
                                                  delegate=mp.tasks.BaseOptions.Delegate.CPU),
                running_mode=mp.tasks.vision.RunningMode.IMAGE if image_mode else mp.tasks.vision.RunningMode.VIDEO,
                num_hands=num_hands, min_hand_detection_confidence=detection_confidence,
                min_hand_presence_confidence=presence_confidence, min_tracking_confidence=.65))
        self.timestamp = -1

    def detect_all(self, image, timestamp):
        import cv2
        import numpy as np
        self.timestamp = max(self.timestamp + 1, int(timestamp * 1000))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        if getattr(self, 'image_mode', False):
            result = self.detector.detect(frame)
        else:
            result = self.detector.detect_for_video(frame, self.timestamp)
        height, width = image.shape[:2]
        hands = []
        for landmarks, world_landmarks, categories in zip(result.hand_landmarks, result.hand_world_landmarks, result.handedness):
            pixels = np.array([(p.x * width, p.y * height) for p in landmarks])
            world = [(p.x, p.y, p.z) for p in world_landmarks]
            category = categories[0]
            normalized = [(p.x, p.y, p.z) for p in landmarks]
            hands.append(Hand(pixels, world, category.category_name, classify(world), category.score, normalized))
        return hands

    def detect(self, image, timestamp):
        """Compatibility for the single-hand diagnostic experiments."""
        hands = self.detect_all(image, timestamp)
        return hands[0] if hands else None

    def close(self):
        self.detector.close()
