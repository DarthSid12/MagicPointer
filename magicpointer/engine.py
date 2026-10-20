"""Deterministic, timestamp-driven gesture engine. Never touches the OS."""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple

Point = Tuple[float, float, float]


class Gesture(str, Enum):
    FIST = "fist"
    POINT = "point"
    LEFT = "left click"
    RIGHT = "right click"
    SCROLL = "scroll"
    PALM = "open palm"
    UNKNOWN = "unrecognized"


@dataclass(frozen=True)
class Observation:
    gesture: Gesture
    wrist: Point
    tip: Point


@dataclass(frozen=True)
class Event:
    kind: str
    value: object = None


class Engine:
    """Coordinates use camera X right/Y down; workspace dimensions use metres.

    Mono preview supplies coordinates normalized to activation palm size instead.
    A chord is latched until POINT/FIST, preventing left->right->left cascades.
    """

    def __init__(self, width=0.40, height=0.25, mirror_x=True,
                 dwell=0.18, activation_dwell=0.45, loss_timeout=0.30):
        values = (width, height, dwell, activation_dwell, loss_timeout)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Workspace sizes and timing thresholds must be positive and finite")
        self.width, self.height, self.mirror_x = width, height, mirror_x
        self.dwell, self.activation_dwell = dwell, activation_dwell
        self.loss_timeout = loss_timeout
        self.active = False
        self.origin = None
        self.cursor = (0.5, 0.5)
        self._candidate = None
        self._since = 0.0
        self._last_time = None
        self._last_seen = None
        self._action_latched = False
        self._scroll_y = None
        self._last_move = None

    def _stop(self, reason):
        self.active = False
        self.origin = None
        self._candidate = None
        self._action_latched = False
        self._scroll_y = None
        self._last_move = None
        return [Event("deactivate", reason)]

    def step(self, observation: Optional[Observation], now: float):
        if not math.isfinite(now) or (self._last_time is not None and now < self._last_time):
            raise ValueError("Timestamps must be finite and monotonic")
        dt = 0.0 if self._last_time is None else now - self._last_time
        self._last_time = now
        valid = observation is not None and all(
            math.isfinite(v) for point in (observation.wrist, observation.tip) for v in point)
        # A long processing stall must not resume a previously armed click.
        if self.active and self._last_seen is not None and now - self._last_seen > self.loss_timeout:
            events = self._stop("tracking lost")
            self._last_seen = now if valid else self._last_seen
            return events
        if not valid:
            self._candidate = None
            self._scroll_y = None
            return []
        self._last_seen = now
        gesture = observation.gesture
        if gesture != self._candidate:
            self._candidate, self._since = gesture, now
            self._scroll_y = None
        elapsed = now - self._since
        if not self.active:
            if gesture == Gesture.FIST and elapsed >= self.activation_dwell:
                self.active = True
                self.origin = observation.wrist
                self.cursor = (0.5, 0.5)
                self._action_latched = False
                return [Event("activate"), Event("move", self.cursor)]
            return []
        if gesture in (Gesture.FIST, Gesture.POINT) and elapsed >= self.dwell:
            self._action_latched = False
        events = []
        if elapsed < self.dwell:
            return events
        return events
