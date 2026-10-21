"""Track local hand identities and bind one matched hand across camera views."""
from dataclasses import dataclass
import itertools
import math

import numpy as np

from .engine import Gesture
from .fusion import combine


class LocalHands:
    """Conservative spatial continuity; detector list order never identifies a hand."""
    def __init__(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, hands, now, size, keep=None):
        self.tracks = {key: value for key, value in self.tracks.items()
                       if now-value[3] <= 1. or key == keep}
        diagonal = math.hypot(*size)
        current = []
        for hand in hands:
            pixels = np.asarray(hand.pixels)
            current.append((hand, pixels[0]/diagonal,
                            max(.008, float(np.linalg.norm(pixels[0]-pixels[9]))/diagonal)))
        costs = {}
        for key, (old, wrist, scale, seen) in self.tracks.items():
            for index, (hand, position, length) in enumerate(current):
                if old.label != hand.label:
                    continue
                cost = float(np.linalg.norm(position-wrist))/max(scale, length) + .3*abs(math.log(length/scale))
                if cost <= 3.:
                    costs[key, index] = cost
        assigned = {}
        used = set()
        for key in self.tracks:
            choices = sorted((cost, index) for (track, index), cost in costs.items() if track == key)
            if not choices:
                continue
            cost, index = choices[0]
            rivals = sorted((value, track) for (track, detection), value in costs.items() if detection == index)
            if rivals[0][1] != key or (len(choices) > 1 and choices[1][0]-cost < .5) or (len(rivals) > 1 and rivals[1][0]-cost < .5):
                continue
            assigned[key] = index
            used.add(index)
        for index in range(len(current)):
            if index not in used and not any(detection == index for _, detection in costs):
                assigned[self.next_id] = index
                self.next_id += 1
        visible = {}
        for key, index in assigned.items():
            hand, wrist, scale = current[index]
            self.tracks[key] = (hand, wrist, scale, now)
            visible[key] = hand
        return visible


@dataclass
class Selection:
    key: object = None
    hands: object = None
    combined: object = None
    reason: str = 'Show one fist to select a hand'
    progress: float = 0.
    spread: object = None


class HandOwner:
    def __init__(self, cameras, dwell=2.5):
        self.locals = [LocalHands() for _ in range(cameras)]
        self.visible = [{} for _ in range(cameras)]
        self.owner = self.candidate = self.since = None
        self.dwell = dwell
        self.palm_since = None
        self.release_ready = False

    def release(self):
        self.owner = self.candidate = self.since = None
        self.palm_since = None
        self.release_ready = False

    def lock(self, key):
        self.owner = key
        self.candidate = self.since = None
        self.palm_since = None
        self.release_ready = False

    def select(self, detections, frames, now, *, stereo=None, scrolling=False):
        self.visible = [tracker.update(hands, now, frame.image.shape[1::-1],
                                      self.owner[index] if self.owner else None)
                        for index, (tracker, hands, frame) in enumerate(zip(self.locals, detections, frames))]
        def result(key):
            hands = [view.get(track) for view, track in zip(self.visible, key)]
            fused, reason, spread = combine(hands, ignore_thumb=scrolling)
            if fused is not None and stereo:
                anchors = (0, 5, 9, 13, 17)
                points = stereo.reconstruct(hands[0].pixels[list(anchors)], hands[1].pixels[list(anchors)])
                if points is None:
                    fused, reason = None, 'Hands do not form a valid stereo pair'
            return Selection(key, hands, fused, reason, spread=spread)
        if self.owner:
            selected = result(self.owner)
            if selected.combined is None:
                selected.reason = 'OWNER PAUSED: '+selected.reason
            if selected.combined is not None and selected.combined.gesture == Gesture.PALM:
                if self.palm_since is None:
                    self.palm_since = now
                self.release_ready = now-self.palm_since >= .18
            else:
                self.palm_since, self.release_ready = None, False
            return selected
        candidates = [result(key) for key in itertools.product(*(view.keys() for view in self.visible))]
        candidates = [item for item in candidates if item.combined is not None]
        # Uncalibrated association needs a unique reciprocal shape match. Similar
        # same-handed hands cannot be disambiguated reliably; wait, never guess.
        if len(self.locals) == 2:
            accepted = []
            for item in candidates:
                unique = True
                for camera in (0, 1):
                    rivals = sorted(other.spread for other in candidates
                                    if other.key[camera] == item.key[camera] and other.key != item.key)
                    if rivals and rivals[0]-item.spread < .08:
                        unique = False
                if unique:
                    accepted.append(item)
            candidates = accepted
        fists = [item for item in candidates if item.combined.gesture == Gesture.FIST]
        if len(fists) != 1:
            self.candidate = self.since = None
            if len(fists) > 1:
                return Selection(reason='MULTIPLE FISTS: keep only one fist raised')
            if len(candidates) == 1:
                item = candidates[0]
                item.reason = 'Show a fist for 2.5 seconds to select this hand'
                return item
            return Selection(reason='Show one clear fist in all views; separate overlapping hands')
        item = fists[0]
        if item.key != self.candidate:
            self.candidate, self.since = item.key, now
        item.progress = min(1., (now-self.since)/self.dwell)
        item.reason = f'HOLD FIST: {max(0., self.dwell-(now-self.since)):.1f}s'
        return item
