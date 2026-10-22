"""Relative image-space motion when calibrated stereo depth is unavailable."""
import numpy as np


class ApproximateMotion:
    def __init__(self):
        self.source = None
        self.anchor = None
        self.base = np.zeros(3)
        self.scale = 1.

    def map(self, hand, source, reference=None):
        changed = self.source != source
        if changed:
            self.source = source
            self.anchor = np.asarray(hand.pixels[0], dtype=float).copy()
            self.base = np.asarray(reference if reference is not None else (0., 0., 0.), dtype=float)
            self.scale = max(10., float(np.linalg.norm(hand.pixels[0]-hand.pixels[9])))
        # A palm-length movement maps to 10 cm in the interaction workspace.
        # These are relative control coordinates, never measured position/depth.
        points = (np.asarray(hand.pixels)-self.anchor)/self.scale*.10
        points = np.column_stack((points, np.zeros(len(points)))) + self.base
        return points, changed

    def reset(self):
        self.source = None
