"""Deterministic, timestamp-driven gesture engine. Never touches the OS."""

from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple

from .scrolling import PitchScroll

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
    scroll_pitch: Optional[float] = None
    point_shape: bool = False


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
    Missing or unrecognized frames shorter than ``grace`` are bridged: they neither
    restart nor count toward gesture dwell, and do not pause. Click poses hold the
    pointer, and a click lands where it was ``click_lookback`` seconds before the
    pose began.
    """

    def __init__(self, width=0.40, height=0.25, mirror_x=True,
                 dwell=0.18, activation_dwell=0.45, loss_timeout=0.30,
                 retain_activation_on_loss=False, activation_grace=0.,
                 grace=0.10, click_lookback=0.10, min_cutoff=1.0, beta=10.0):
        values = (width, height, dwell, activation_dwell, loss_timeout, grace, click_lookback, min_cutoff, beta)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("Workspace sizes and timing thresholds must be positive and finite")
        self.width, self.height, self.mirror_x = width, height, mirror_x
        self.dwell, self.activation_dwell = dwell, activation_dwell
        self.grace, self.click_lookback = grace, click_lookback
        self._filter = OneEuro(min_cutoff, beta)
        self._history = deque(maxlen=64)
        self._candidate_seen = 0.0
        self._interrupted = self._bridged = False
        self._previous_frame = None
        if not math.isfinite(activation_grace) or activation_grace < 0:
            raise ValueError("Activation grace must be finite and nonnegative")
        self.loss_timeout = loss_timeout
        self.retain_activation_on_loss = retain_activation_on_loss
        self.activation_grace = activation_grace
        self.paused = False
        self.reset_activation()
        self.active = False
        self.origin = None
        self.cursor = (0.5, 0.5)
        self._candidate = None
        self._since = 0.0
        self._last_time = None
        self._last_seen = None
        self._action_latched = False
        self.scrolling = False
        self.scroll = PitchScroll()
        self._exit_since = None
        self._offset = (0., 0.)
        self._rebase = False
        self._last_move = None

    def reset_activation(self):
        self._fist_seconds = 0.
        self._last_fist = None
        self._previous_fist = False

    @property
    def activation_progress(self):
        return 1. if self.active else min(1., self._fist_seconds/self.activation_dwell)

    def diagnostics(self, now):
        return {'active': self.active, 'paused': self.paused,
                'gesture_candidate': self._candidate, 'candidate_since': self._since,
                'last_seen': self._last_seen,
                'missing_seconds': None if self._last_seen is None else now-self._last_seen,
                'fist_valid_seconds': self._fist_seconds, 'last_fist': self._last_fist,
                'click_latched': self._action_latched, 'scrolling': self.scrolling,
                'rebase_pending': self._rebase}

    def _pause(self):
        if not self.paused:
            self._candidate = None
            self._action_latched = True
            self._rebase = True
            self._last_move = self._exit_since = None
            self.scroll.reset()
        self.paused = True

    def stop(self, reason):
        self.active = False
        self.paused = False
        self.reset_activation()
        self.origin = None
        self._candidate = None
        self._action_latched = False
        self.scrolling = False
        self.scroll = PitchScroll()
        self._exit_since = None
        self._offset = (0., 0.)
        self._rebase = False
        self._last_move = None
        self._history.clear()
        return [Event("deactivate", reason)]

    def _settle(self, gesture, now):
        """Return the pose to act on, bridging brief recognition dropouts.

        Unrecognized or missing frames within the grace period neither restart
        nor count toward the current pose's dwell, and actions only fire on
        frames where the pose was actually seen. So one noisy frame cannot
        restart a click, and a spurious pose cannot accumulate hold time through
        misreads. A different recognized pose always restarts dwell.
        """
        # Only an interruption expires the candidate, so slow frame rates still dwell.
        if self._interrupted and self._candidate is not None and now - self._candidate_seen > self.grace:
            self._candidate = None
        if gesture == Gesture.UNKNOWN and self._candidate not in (None, Gesture.UNKNOWN):
            self._interrupted = self._bridged = True
            return self._candidate
        self._bridged = False
        if gesture != self._candidate:
            self._candidate, self._since = gesture, now
        elif self._interrupted and self._previous_frame is not None:
            self._since += max(0., self._previous_frame - self._candidate_seen)
        self._candidate_seen, self._interrupted = now, False
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
        dt = 0.0 if self._last_time is None else now - self._last_time
        self._previous_frame, self._last_time = self._last_time, now
        valid = observation is not None and all(
            math.isfinite(v) for point in (observation.wrist, observation.tip) for v in point)
        # A selected owner pauses through gaps; legacy stand-alone engines can
        # still deactivate. Recovery never reuses a pending click or scroll rate.
        if self.active and self.retain_activation_on_loss:
            late = self._last_seen is not None and now-self._last_seen > self.loss_timeout
            if not valid and not late and self._last_seen is not None and now-self._last_seen <= self.grace:
                # Bridged: neither pause nor count this gap toward any dwell.
                self._interrupted = True
                return []
            if not valid or late:
                self._pause()
                if valid:
                    self._last_seen = now
                return []
            if self.paused:
                self.paused = False
        if self.active and self._last_seen is not None and now-self._last_seen > self.loss_timeout:
            events = self.stop("tracking lost")
            self._last_seen = now if valid else self._last_seen
            return events
        if not self.active and self.activation_grace > 0:
            gesture = observation.gesture if valid else Gesture.UNKNOWN
            if gesture == Gesture.FIST:
                if self._last_fist is None or now-self._last_fist > self.activation_grace:
                    self.reset_activation()
                if self._previous_fist:
                    self._fist_seconds += dt
                self._last_fist, self._previous_fist = now, True
                if self._fist_seconds >= self.activation_dwell:
                    self.active = True
                    self.origin, self.cursor = observation.wrist, (.5, .5)
                    self._last_seen = now
                    self._candidate, self._since = gesture, now
                    self._action_latched = False
                    return [Event("activate"), Event("move", self.cursor)]
            elif gesture == Gesture.UNKNOWN:
                self._previous_fist = False
                if self._last_fist is not None and now-self._last_fist > self.activation_grace:
                    self.reset_activation()
            else:
                self.reset_activation()
            return []
        if not valid:
            # Brief gaps are bridged; the candidate expires in _settle after the grace period.
            self._interrupted = True
            if self._last_seen is None or now-self._last_seen > self.grace:
                self._candidate = None
                self.scroll.update(None, now)
                self._exit_since = None
            return []
        self._last_seen = now
        gesture = self._settle(observation.gesture, now)
        elapsed = now - self._since
        if not self.active:
            if gesture == Gesture.FIST and elapsed >= self.activation_dwell and not self._bridged:
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
            self.scroll.update(None, now)
            self._exit_since = None
            return self.stop("open palm") if elapsed >= self.dwell and not self._bridged else []
        if (not self.scrolling and gesture == Gesture.SCROLL and elapsed >= self.dwell and not self._bridged
                and not self._action_latched and observation.scroll_pitch is not None):
            self.scrolling = True
            self.scroll.reset()
        if self.scrolling:
            # Thumb is deliberately irrelevant once latched. Only a stable
            # index-only shape/fist releases the mode; a palm stops globally.
            exiting = (observation.scroll_pitch is None and
                       (observation.point_shape or gesture in (Gesture.POINT, Gesture.FIST)))
            if exiting:
                self.scroll.update(None, now)
                if self._exit_since is None:
                    self._exit_since = now
                if now-self._exit_since >= self.dwell:
                    self.scrolling = False
                    self.scroll.reset()
                    self._rebase = True
                    self._last_move = None
                    self._action_latched = gesture not in (Gesture.POINT, Gesture.FIST)
                return []
            self._exit_since = None
            lines = self.scroll.update(observation.scroll_pitch, now)
            return [Event("scroll", lines)] if lines else []
        if gesture in (Gesture.FIST, Gesture.POINT) and elapsed >= self.dwell and not self._bridged:
            self._action_latched = False
        events = []
        if gesture == Gesture.POINT:
            x = (observation.tip[0] - self.origin[0]) / self.width
            y = (observation.tip[1] - self.origin[1]) / self.height
            raw = (.5 + (-x if self.mirror_x else x), .5 + y)
            if self._rebase:
                self._offset = tuple(a-b for a, b in zip(self.cursor, raw))
                self._rebase = False
            target = tuple(max(0., min(1., value+offset)) for value, offset in zip(raw, self._offset))
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
        if elapsed < self.dwell or self._bridged:
            return events
        if gesture in (Gesture.LEFT, Gesture.RIGHT) and not self._action_latched:
            self._action_latched = True
            # Opening the thumb drags the index tip; click where the pointer was before.
            anchor = self._before(self._since - self.click_lookback)
            if anchor != self.cursor:
                self.cursor = anchor
                self._filter.reset(anchor)
                events.append(Event("move", anchor))
            events.append(Event("click", "left" if gesture == Gesture.LEFT else "right"))
        return events
