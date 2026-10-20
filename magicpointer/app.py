import cv2
import numpy as np
from .cameras import Cameras
from .tracking import Tracker, Gesture

COLORS = {"inactive": (150, 150, 150), "unavailable": (0, 160, 255)}

def draw_hand(image, hand, color):
    if hand is None:
        return
    points = hand.pixels.astype(int)
    x, y = points.min(axis=0) - 15
    x2, y2 = points.max(axis=0) + 15
    cv2.rectangle(image, (int(x), int(y)), (int(x2), int(y2)), color, 3)
    for start in (1, 5, 9, 13, 17):
        cv2.line(image, tuple(points[0]), tuple(points[start]), color, 2)
        for index in range(start, start+3):
            cv2.line(image, tuple(points[index]), tuple(points[index+1]), color, 2)
    for point in points:
        cv2.circle(image, tuple(point), 4, color, -1)


def run(args):
    if not args.debug:
        raise ValueError("This checkpoint supports run --debug; activation arrives in feature 4")
    tracker = None
    window = "MagicPointer debug - Q / Escape stops"
    try:
        tracker = Tracker(args.model)
        with Cameras(args.left, args.right, args.fps) as cameras:
            print("Camera negotiated settings:", cameras.info)
            while True:
                frame = cameras.read()[0]
                hand = tracker.detect(frame.image, frame.grabbed_at)
                color = COLORS['inactive' if hand else 'unavailable']
                panel = frame.image.copy()
                draw_hand(panel, hand, color)
                panel = cv2.resize(panel, (960, 540))
                label = f"DEBUG / NO DESKTOP INPUT | inactive | {hand.gesture.value if hand else 'no hand'}"
                cv2.putText(panel, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 2)
                cv2.imshow(window, panel)
                if (cv2.waitKey(1) & 255) in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return 0
    finally:
        if tracker:
            tracker.close()
        cv2.destroyAllWindows()
