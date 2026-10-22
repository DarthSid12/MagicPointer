"""Deterministic, timestamp-driven gesture engine. Never touches the OS."""

from collections import deque
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


class OneEuro:
    """One-Euro filter (Casiez et al., 2012): smooths jitter at rest, little lag in motion."""

    def __init__(self, min_cutoff, beta, speed_cutoff=1.0):
        self.min_cutoff, self.beta, self.speed_cutoff = min_cutoff, beta, speed_cutoff
        self.value, self.speed = None, None

    def reset(self, value):
        self.value, self.speed = tuple(value), (0.,) * len(value)

    @staticmethod
    def _alpha(dt, cutoff):
        return 1 / (1 + 1 / (2 * math.pi * cutoff * dt))

    def __call__(self, value, dt):
        if self.value is None:
            self.reset(value)
        elif dt > 0:
            a = self._alpha(dt, self.speed_cutoff)
            self.speed = tuple(s + a * ((v - x) / dt - s) for s, v, x in zip(self.speed, value, self.value))
            a = self._alpha(dt, self.min_cutoff + self.beta * math.hypot(*self.speed))
            self.value = tuple(x + a * (v - x) for x, v in zip(self.value, value))
        return self.value


class Engine:
    """Coordinates use camera X right/Y down; workspace dimensions use metres.

    Mono preview supplies coordinates normalized to activation palm size instead.
    A chord is latched until POINT/FIST, preventing left->right->left cascades.
    Recognition dropouts shorter than ``grace`` are bridged rather than restarting
    dwell timers. Click poses hold the pointer, and the click lands where it was
    ``click_lookback`` seconds before the pose began, before the thumb dragged the tip.
    """

    def __init__(self, width=0.40, height=0.25, mirror_x=True,
                 dwell=0.18, activation_dwell=0.45, loss_timeout=0.30, grace=0.10,
                 click_lookback=0.10, min_cutoff=1.0, beta=10.0):
        values = (width, height, dwell, activation_dwell, loss_timeout, grace,
                  click_lookback, min_cutoff, beta)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Workspace sizes and timing thresholds must be positive and finite")
        self.width, self.height, self.mirror_x = width, height, mirror_x
        self.dwell, self.activation_dwell = dwell, activation_dwell
        self.loss_timeout, self.grace, self.click_lookback = loss_timeout, grace, click_lookback
        self.active = False
        self.origin = None
        self.cursor = (0.5, 0.5)
        self._filter = OneEuro(min_cutoff, beta)
        self._history = deque(maxlen=64)
        self._candidate = None
        self._since = 0.0
        self._candidate_seen = 0.0
        self._previous = None
        self._last_time = None
        self._last_seen = None
        self._action_latched = False
        self._scroll_y = None
        self._last_move = None

    def _stop(self, reason):
        self.active = False
        self.origin = None
        self._candidate = None
        self._previous = None
        self._action_latched = False
        self._scroll_y = None
        self._last_move = None
        self._history.clear()
        return [Event("deactivate", reason)]

    def _settle(self, gesture, now):
        """Return the pose to act on, bridging brief recognition dropouts.

        An unrecognized frame within the grace period continues the current pose,
        and returning to a pose interrupted for no longer than the grace period
        resumes its dwell timer, so one noisy frame cannot restart activation or a click.
        """
        if self._candidate is not None and now - self._candidate_seen > self.grace:
            self._candidate, self._previous, self._scroll_y = None, None, None
        if gesture == Gesture.UNKNOWN and self._candidate not in (None, Gesture.UNKNOWN):
            return self._candidate
        if gesture != self._candidate:
            previous = self._previous
            if self._candidate is not None:
                self._previous = (self._candidate, self._since, self._candidate_seen)
            resume = previous is not None and previous[0] == gesture and now - previous[2] <= self.grace
            self._candidate, self._since = gesture, previous[1] if resume else now
            self._scroll_y = None
        self._candidate_seen = now
        return gesture

    def _before(self, when):
        """Pointer position at or just before ``when``; the oldest kept if none is that old."""
        position = self.cursor
        for moment, cursor in reversed(self._history):
            position = cursor
            if moment <= when:
                break
        return position

    def step(self, observation: Optional[Observation], now: float):
        if not math.isfinite(now) or (self._last_time is not None and now < self._last_time):
            raise ValueError("Timestamps must be finite and monotonic")
        self._last_time = now
        valid = observation is not None and all(
            math.isfinite(v) for point in (observation.wrist, observation.tip) for v in point)
        # A long processing stall must not resume a previously armed click.
        if self.active and self._last_seen is not None and now - self._last_seen > self.loss_timeout:
            events = self._stop("tracking lost")
            self._last_seen = now if valid else self._last_seen
            return events
        if not valid:
            # Candidate expiry happens in _settle once the gap exceeds the grace period.
            return []
        self._last_seen = now
        gesture = self._settle(observation.gesture, now)
        elapsed = now - self._since
        if not self.active:
            if gesture == Gesture.FIST and elapsed >= self.activation_dwell:
                self.active = True
                self.origin = observation.wrist
                self.cursor = (0.5, 0.5)
                self._action_latched = False
                self._last_move = None
                self._history.clear()
                return [Event("activate"), Event("move", self.cursor)]
            return []
        # Stop motion immediately on a palm candidate, then confirm the transition.
        if gesture == Gesture.PALM:
            return self._stop("open palm") if elapsed >= self.dwell else []
        if gesture in (Gesture.FIST, Gesture.POINT) and elapsed >= self.dwell:
            self._action_latched = False
        events = []
        if gesture == Gesture.POINT:
            x = (observation.tip[0] - self.origin[0]) / self.width
            y = (observation.tip[1] - self.origin[1]) / self.height
            target = (max(0., min(1., .5 + (-x if self.mirror_x else x))),
                      max(0., min(1., .5 + y)))
            # Time-based filtering behaves similarly at 30 and 60 Hz.
            if self._last_move is None:
                self._filter.reset(target)
            self.cursor = self._filter(target, 0. if self._last_move is None else now - self._last_move)
            self._last_move = now
            self._history.append((now, self.cursor))
            events.append(Event("move", self.cursor))
        elif gesture in (Gesture.LEFT, Gesture.RIGHT):
            # Hold still while a click pose forms; resume pointing from here.
            self._filter.reset(self.cursor)
        if elapsed < self.dwell:
            return events
        if gesture in (Gesture.LEFT, Gesture.RIGHT) and not self._action_latched:
            self._action_latched = True
            anchor = self._before(self._since - self.click_lookback)
            if anchor != self.cursor:
                self.cursor = anchor
                self._filter.reset(anchor)
                events.append(Event("move", anchor))
            events.append(Event("click", "left" if gesture == Gesture.LEFT else "right"))
        elif gesture == Gesture.SCROLL and not self._action_latched:
            # Entry emits nothing; all motion is relative to the entry location.
            y = observation.tip[1]
            if self._scroll_y is None:
                self._scroll_y = y
            else:
                lines = int((self._scroll_y - y) / self.height * 30)
                if lines:
                    self._scroll_y -= lines * self.height / 30
                    events.append(Event("scroll", max(-10, min(10, lines))))
        return events
