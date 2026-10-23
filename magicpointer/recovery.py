"""Re-detect a recently seen hand in a magnified crop around its last position.

The palm detector sees each full frame shrunk to about 192 pixels, so a hand a
few metres from a 1080p camera is only a few pixels wide and is easily lost.
Searching a crop around the hand's last box magnifies it roughly tenfold.
Recovered hands keep the normal detection thresholds and still have to pass
ownership matching, cross-view shape agreement and stereo checks.
"""
from dataclasses import dataclass, replace

import numpy as np


@dataclass
class Region:
    box: tuple  # x0, y0, x1, y1 in full-frame pixels
    seen: float


def box_of(hand):
    points = np.asarray(hand.pixels, dtype=float)
    return (*points.min(axis=0), *points.max(axis=0))


def duplicates(hand, others):
    """True when another detection's wrist is within half a hand box of this one."""
    wrist = np.asarray(hand.pixels, dtype=float)[0]
    x0, y0, x1, y1 = box_of(hand)
    limit = .5 * max(x1 - x0, y1 - y0)
    return any(np.linalg.norm(np.asarray(other.pixels, dtype=float)[0] - wrist) < limit for other in others)


def contains(box, hand, margin=0.):
    x0, y0, x1, y1 = box
    wrist = np.asarray(hand.pixels, dtype=float)[0]
    pad = margin * max(x1 - x0, y1 - y0)
    return x0 - pad <= wrist[0] <= x1 + pad and y0 - pad <= wrist[1] <= y1 + pad


class CropRecovery:
    memory = 1.5  # seconds a lost hand's region is still searched
    scale = 3.0   # crop edge as a multiple of the hand box's longer side
    minimum = 256

    def __init__(self, factory, views):
        self.factory = factory
        self.trackers = [None] * views
        self.regions = [None] * views
        self.recovered = [0] * views

    def remember(self, view, hand, now):
        if hand is not None and len(getattr(hand, 'pixels', ())) == 21 and np.isfinite(hand.pixels).all():
            self.regions[view] = Region(box_of(hand), now)

    def crop(self, image, box):
        height, width = image.shape[:2]
        x0, y0, x1, y1 = box
        side = int(min(max(self.minimum, self.scale * max(x1 - x0, y1 - y0)), width, height))
        left = int(min(max(0, (x0 + x1) / 2 - side / 2), width - side))
        top = int(min(max(0, (y0 + y1) / 2 - side / 2), height - side))
        return image[top:top + side, left:left + side], left, top

    def _search(self, view, frame, region):
        image = frame.image
        crop, left, top = self.crop(image, region.box)
        if self.trackers[view] is None:
            self.trackers[view] = self.factory()
        height, width = image.shape[:2]
        found = []
        for hand in self.trackers[view].detect_all(crop, frame.grabbed_at):
            pixels = np.asarray(hand.pixels, dtype=float) + [left, top]
            normalized = None
            if hand.image_landmarks is not None:
                normalized = [((left + x * crop.shape[1]) / width, (top + y * crop.shape[0]) / height, z)
                              for x, y, z in hand.image_landmarks]
            mapped = replace(hand, pixels=pixels, image_landmarks=normalized)
            # Only accept a hand that is still near where the lost one was.
            if contains(region.box, mapped, margin=1.):
                found.append(mapped)
        return found

    def apply(self, detections, frames, now, pool=None):
        """Detections with crop-recovered hands added to views that lost a recent hand."""
        jobs = []
        for view, (hands, frame) in enumerate(zip(detections, frames)):
            region = self.regions[view]
            if region is None or now - region.seen > self.memory:
                continue
            if any(contains(region.box, hand, margin=.5) for hand in hands):
                continue
            jobs.append((view, frame, region))
        if not jobs:
            return detections
        search = lambda job: self._search(*job)
        results = list(pool.map(search, jobs)) if pool and len(jobs) > 1 else [search(job) for job in jobs]
        detections = list(detections)
        for (view, _, _), found in zip(jobs, results):
            # A hand the regular detector already found elsewhere is not added twice.
            found = [hand for hand in found if not duplicates(hand, detections[view])]
            self.recovered[view] += bool(found)
            detections[view] = list(detections[view]) + found
        return detections

    def close(self):
        for tracker in self.trackers:
            if tracker is not None:
                tracker.close()
