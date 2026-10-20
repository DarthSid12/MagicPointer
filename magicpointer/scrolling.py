"""Automatic neutral capture and time-based wrist scrolling; no camera or OS IO."""
from collections import deque
import math


def wrapped(angle):
    return (angle + 180.) % 360. - 180.


def tilt_speed(delta, dead_zone=10., full_speed_angle=30.):
    """Positive is downward; matches the wrist experiment's bounded response."""
    return math.copysign(min(1., max(0., abs(delta)-dead_zone) /
                             (full_speed_angle-dead_zone))*12., delta)


class PitchScroll:
    def __init__(self):
        self.reset()

    def reset(self):
        self.history = deque()
        self.neutral = self.filtered = self.last_valid = self.last_time = None
        self.delta = None
        self.speed = 0.
        self.fraction = 0.

    def update(self, pitch, now):
        dt = 0. if self.last_time is None else now-self.last_time
        if self.last_valid is not None and now-self.last_valid > .30:
            self.reset()
        self.last_time = now
        if pitch is None or not math.isfinite(pitch):
            self.filtered = self.delta = None
            self.history.clear()
            self.speed = self.fraction = 0.
            return 0
        self.last_valid = now
        self.history.append((now, pitch))
        while self.history and now-self.history[0][0] > .45:
            self.history.popleft()
        alpha = 1. if self.filtered is None else 1.-math.exp(-dt/.10)
        self.filtered = pitch if self.filtered is None else wrapped(self.filtered + alpha*wrapped(pitch-self.filtered))
        if self.neutral is None:
            reference = self.history[0][1]
            values = [wrapped(p-reference) for _, p in self.history]
            if len(values) >= 3 and now-self.history[0][0] >= .30 and max(values)-min(values) <= 6.:
                self.neutral = wrapped(reference + sum(values)/len(values))
            return 0
        self.delta = wrapped(self.filtered-self.neutral)
        speed = tilt_speed(self.delta)
        if speed == 0 or speed*self.speed < 0:
            self.fraction = 0.
        self.speed = speed
        # OS wheel events are integral, positive UP. Retain only the fraction,
        # never a backlog after tracking loss or a long processing stall.
        self.fraction -= speed*min(dt, .10)
        lines = math.trunc(self.fraction)
        self.fraction -= lines
        return lines
