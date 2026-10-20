"""Palm-relative finger pitch and a preview-only calibrated speed controller."""
from collections import deque
import math

import numpy as np

from .tracking import gesture_metrics


def wrapped(angle):
    return (angle + 180.) % 360. - 180.


def measure(hand):
    if hand is None:
        return None, 'No hand'
    try:
        points = np.asarray(hand.world, dtype=float)
    except (ValueError, TypeError):
        return None, 'Invalid landmarks'
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return None, 'Invalid landmarks'
    metrics = gesture_metrics(points)
    if not metrics['valid']:
        return None, 'Invalid palm'
    fingers = metrics['fingers']
    if metrics['thumb']['state'] != 0 or any(fingers[name]['state'] != 0 for name in ('ring', 'little')):
        return None, 'Tuck thumb and curl ring/little fingers'
    # MCP flexion is the experimental signal. The normal classifier also checks
    # reach from the wrist, so it may reject a valid tilted pair. Keep straight
    # finger chains here, without requiring unchanged wrist-to-tip reach.
    if any(fingers[name]['straightness'] <= .8 for name in ('index', 'middle')):
        return None, 'Keep index and middle fingers extended'
    across = points[5] - points[17]
    length = np.linalg.norm(across)
    if length < 1e-6:
        return None, 'Palm geometry unavailable'
    across /= length
    forward = points[9] - points[0]
    forward -= across * np.dot(forward, across)
    length = np.linalg.norm(forward)
    if length < 1e-6:
        return None, 'Palm geometry unavailable'
    forward /= length
    normal = np.cross(across, forward)
    axes = []
    for base in (5, 9):
        axis = points[base+3] - points[base]
        length = np.linalg.norm(axis)
        if length < 1e-6:
            return None, 'Finger geometry unavailable'
        axes.append(axis/length)
    if np.dot(*axes) < .75:
        return None, 'Point both fingers in the same direction'
    axis = axes[0] + axes[1]
    x, y = float(np.dot(axis, forward)), float(np.dot(axis, normal))
    if math.hypot(x, y) < .5:
        return None, 'Fingers are too far sideways for pitch measurement'
    return math.degrees(math.atan2(y, x)), 'Two-finger shape valid'


def measure_wrist(hand):
    """Camera-relative finger direction: image Y down, negative Z toward camera.

    This is a model estimate from one camera, not calibrated stereo orientation.
    Shape validation is shared, but palm-relative pitch is not the control signal.
    """
    valid_pitch, reason = measure(hand)
    if valid_pitch is None:
        return None, reason
    points = np.asarray(hand.world, dtype=float)
    axes = [points[base+3]-points[base] for base in (5, 9)]
    axis = sum(vector/np.linalg.norm(vector) for vector in axes)
    axis /= np.linalg.norm(axis)
    if -axis[2] < .25:
        return None, 'Point the two fingers toward the screen/camera'
    pitch = math.degrees(math.atan2(float(axis[1]), math.hypot(float(axis[0]), float(axis[2]))))
    return pitch, 'Two-finger direction valid'


class Tilt:
    def __init__(self, dead_zone=5.):
        if not math.isfinite(dead_zone) or not 0 < dead_zone < 30:
            raise ValueError('Tilt dead zone must be between 0 and 30 degrees')
        self.dead_zone = dead_zone
        self.history = deque()
        self.neutral = self.span = self.direction = None
        self.filtered = self.current = self.label = None
        self.last_valid = self.last_time = None

    def reset(self):
        self.neutral = self.span = self.direction = None
        self.filtered = self.current = self.label = None
        self.history.clear()

    def update(self, hand, now):
        if not math.isfinite(now) or (self.last_time is not None and now < self.last_time):
            raise ValueError('Tilt timestamps must be finite and monotonic')
        dt = 0. if self.last_time is None else now-self.last_time
        self.last_time = now
        if self.last_valid is not None and now-self.last_valid > .30:
            self.reset()
        if hand is not None and self.label is not None and hand.label != self.label:
            self.reset()
        pitch, reason = self._measure(hand)
        self.current = pitch
        self.current_label = hand.label if hand is not None else None
        if pitch is None:
            self.filtered = None
            self.history.clear()
            return self._result(False, reason, None, 0.)
        self.last_valid = now
        self.history.append((now, pitch))
        while self.history and now-self.history[0][0] > .45:
            self.history.popleft()
        alpha = 1. if self.filtered is None else 1.-math.exp(-dt/.10)
        self.filtered = pitch if self.filtered is None else wrapped(self.filtered + alpha*wrapped(pitch-self.filtered))
        delta, speed = None, 0.
        if self.neutral is not None:
            delta = wrapped(self.filtered-self.neutral)
            if self.direction is not None:
                delta *= self.direction
                magnitude = max(0., abs(delta)-self.dead_zone)
                speed = math.copysign(min(1., magnitude/(self.span-self.dead_zone))*12., delta)
        return self._result(True, reason, delta, speed)

    def _result(self, valid, reason, delta, speed):
        return {'valid': valid, 'reason': reason, 'raw_degrees': self.current,
                'dead_zone_degrees': self.dead_zone,
                'smoothed_degrees': self.filtered, 'delta_degrees': delta,
                'lines_per_second': speed, 'phase': 'neutral' if self.neutral is None else ('down' if self.span is None else 'live')}

    def _measure(self, hand):
        return measure(hand)

    def _stable_pitch(self):
        if self.current is None or len(self.history) < 3 or self.history[-1][0]-self.history[0][0] < .30:
            raise ValueError('Hold the two-finger pose steady for at least half a second')
        reference = self.history[0][1]
        values = [wrapped(p-reference) for _, p in self.history]
        if max(values)-min(values) > 6.:
            raise ValueError('Angle is moving too much; hold steady and try again')
        return wrapped(reference + sum(values)/len(values))

    def capture_neutral(self):
        self.neutral = self._stable_pitch()
        self.label = self.current_label
        self.span = self.direction = None

    def capture_down(self):
        if self.neutral is None:
            raise ValueError('Capture neutral with Space first')
        delta = wrapped(self._stable_pitch()-self.neutral)
        if not self.dead_zone+5 <= abs(delta) <= 65:
            raise ValueError(f'Tilt {self.dead_zone+5:g}-65 degrees from neutral, hold steady, then press D')
        self.span, self.direction = abs(delta), math.copysign(1., delta)


class WristTilt(Tilt):
    def __init__(self, dead_zone=10., full_speed_angle=30.):
        super().__init__(dead_zone)
        if not math.isfinite(full_speed_angle) or not dead_zone < full_speed_angle <= 60:
            raise ValueError('Full-speed wrist angle must exceed the dead zone and be at most 60 degrees')
        self.full_speed_angle = full_speed_angle

    def _measure(self, hand):
        return measure_wrist(hand)

    def capture_neutral(self):
        super().capture_neutral()
        self.span, self.direction = self.full_speed_angle, 1.
