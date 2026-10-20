import time

import cv2
import numpy as np

from .audio import Chimes
from .cameras import Cameras
from .engine import Engine, Gesture, Observation
from .stereo import Stereo
from .tracking import Tracker, distance

COLORS = {"inactive": (150, 150, 150), "active": (70, 210, 70),
          "click": (220, 120, 240), "scroll": (240, 180, 30), "unavailable": (0, 160, 255)}


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
    if not args.debug and args.right is None:
        raise ValueError("Real pointer control needs two cameras. For one-camera preview use run --debug")
    stereo = Stereo.load(args.calibration) if args.right is not None else None
    if stereo and stereo.data.get("camera_indices") != [args.left, args.right]:
        raise ValueError("Camera indices/order differ from calibration; select the calibrated pair or recalibrate")
    engine = Engine(width=args.workspace_width * (10 if stereo is None else 1),
                    height=args.workspace_height * (10 if stereo is None else 1), mirror_x=args.mirror_x)
    trackers, audio, desktop = [], None, None
    window = "MagicPointer - Q / Escape stops"
    current_label = None
    mono_scale = 1.
    verified, verification_count = stereo is None, 0
    next_marker_check = 0.
    previous_wrist, previous_wrist_time = None, None
    try:
        # Verify model/dependencies before opening camera devices.
        trackers.append(Tracker(args.model))
        if stereo:
            trackers.append(Tracker(args.model))
        if not args.debug:
            from .desktop import Desktop
            desktop = Desktop()
        audio = Chimes(args.mute)
        with Cameras(args.left, args.right, args.fps) as cameras:
            print("Camera negotiated settings:", cameras.info)
            print("DEBUG: preview only; no OS pointer input" if args.debug else "STEREO CONTROL: Q/Escape or close window to stop")
            print("Hold fist 0.45 s to activate; hold gestures 0.18 s. Return to index-only between clicks.")
            start, count, measured_fps = time.monotonic(), 0, 0.
            last_action, action_until = "", 0.
            while True:
                frames = cameras.read()
                now = time.monotonic()
                paired = stereo.check_frames(frames) if stereo else True
                marker_message = "Mono preview: depth is not measured"
                if stereo and (not verified or now >= next_marker_check):
                    good, marker_message = stereo.check_markers([f.image for f in frames], require_visible=not verified)
                    if not verified:
                        verification_count = verification_count + 1 if good and paired else 0
                        verified = verification_count >= 3
                    elif not good:
                        raise ValueError(marker_message)
                    next_marker_check = now + 1.
                hands = [tracker.detect(frame.image, frame.grabbed_at) for tracker, frame in zip(trackers, frames)]
                observation, reason = None, "No reliable hand"
                left = hands[0]
                if not verified:
                    reason = marker_message
                elif not paired:
                    reason = "Camera pair timing mismatch"
                elif left is not None:
                    if stereo:
                        right = hands[1]
                        if right is not None and left.label == right.label and left.gesture == right.gesture:
                            points = stereo.reconstruct(left.pixels, right.pixels)
                            if points is not None and np.linalg.norm(points[0]) <= 3.6576:
                                observation = Observation(left.gesture, tuple(points[0]), tuple(points[8]))
                            else:
                                reason = "Depth unreliable or beyond 12-foot target range"
                        else:
                            reason = "Hands/gestures disagree between cameras"
                    else:
                        if not engine.active:
                            mono_scale = max(10., float(np.linalg.norm(left.pixels[0]-left.pixels[9])))
                        points = left.pixels / mono_scale
                        observation = Observation(left.gesture, (*points[0], 0.), (*points[8], 0.))
                    if engine.active and left.label != current_label:
                        observation, reason = None, "Controlling hand changed"
                # Reject implausible same-label hand switches in metric stereo mode.
                if observation is not None and stereo and engine.active and previous_wrist is not None:
                    elapsed = now - previous_wrist_time
                    if elapsed > 0 and distance(observation.wrist, previous_wrist) / elapsed > 4.:
                        observation, reason = None, "Hand moved too abruptly; reacquire"
                if observation is not None:
                    previous_wrist, previous_wrist_time = observation.wrist, now
                events = engine.step(observation, time.monotonic())
                for event in events:
                    if event.kind == "activate":
                        current_label = left.label
                        audio.play(True)
                    elif event.kind == "deactivate":
                        current_label, previous_wrist = None, None
                        audio.play(False)
                    elif event.kind == "move" and desktop:
                        desktop.move(event.value)
                    elif event.kind == "click":
                        last_action, action_until = f"{event.value} click", now + .5
                        if desktop:
                            desktop.click(event.value)
                    elif event.kind == "scroll":
                        last_action, action_until = f"scroll {event.value:+d}", now + .3
                        if desktop:
                            desktop.scroll(event.value)
                state = "inactive" if not engine.active else "active"
                if observation is None:
                    state = "unavailable"
                elif engine.active and left.gesture == Gesture.SCROLL:
                    state = "scroll"
                elif now < action_until:
                    state = "click"
                color = COLORS[state]
                if args.debug:
                    panel = frames[0].image.copy()
                    draw_hand(panel, left, color)
                    panel = cv2.resize(panel, (960, 540))
                else:
                    panel = np.full((540, 960, 3), (25, 25, 25), dtype=np.uint8)
                # Primary-camera raw view is intentionally labelled: controls mirror X by default.
                lines = ["DEBUG / NO DESKTOP INPUT" if args.debug else "STEREO / DESKTOP CONTROL",
                         f"State: {state} | Gesture: {left.gesture.value if left else 'none'}",
                         reason if observation is None else (last_action if now < action_until else "Tracking"),
                         f"Processing: {measured_fps:.1f} FPS | Q / Esc / close window to stop"]
                for i, line in enumerate(lines):
                    cv2.putText(panel, line, (15, 27+i*26), cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 0, 0), 4)
                    cv2.putText(panel, line, (15, 27+i*26), cv2.FONT_HERSHEY_SIMPLEX, .58, color, 1)
                for i, (name, value) in enumerate(COLORS.items()):
                    cv2.putText(panel, name, (15+i*185, 522), cv2.FONT_HERSHEY_SIMPLEX, .5, value, 1)
                # Small virtual screen shows motion even when using just one camera.
                cv2.rectangle(panel, (730, 120), (945, 255), (220, 220, 220), 1)
                if engine.active:
                    cv2.circle(panel, (730+round(engine.cursor[0]*215), 120+round(engine.cursor[1]*135)), 5, color, -1)
                cv2.imshow(window, panel)
                key = cv2.waitKey(1) & 255
                if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return 0
                count += 1
                if now - start >= 1:
                    measured_fps, start, count = count/(now-start), now, 0
    finally:
        if audio:
            if engine.active:
                audio.play(False)
            audio.close()
        for tracker in trackers:
            tracker.close()
        cv2.destroyAllWindows()
