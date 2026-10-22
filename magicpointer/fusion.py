"""A single gesture from palm-aligned landmark evidence in matched camera views."""
import numpy as np

from .tracking import Hand, classify


def canonical(hand):
    if hand is None:
        return None
    points = np.asarray(hand.world, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return None
    points = points-points[0]
    scale = np.linalg.norm(points[9])
    across = points[5]-points[17]
    if scale < 1e-6 or np.linalg.norm(across) < 1e-6:
        return None
    across /= np.linalg.norm(across)
    forward = points[9]-across*np.dot(points[9], across)
    if np.linalg.norm(forward) < 1e-6:
        return None
    forward /= np.linalg.norm(forward)
    normal = np.cross(across, forward)
    return points @ np.column_stack((across, forward, normal)) / scale


def combine(hands, *, ignore_thumb=False, bound_pair=False, diagnostics=None):
    """Normalize each view, average geometry, then classify ONCE.

    Labels are diagnostics, not votes. No camera gets to trigger input alone in
    a two-camera session. Canonical coordinates are not metric stereo depth.
    """
    if not hands or any(hand is None for hand in hands):
        return None, 'Selected hand missing from a camera', None
    if not bound_pair and len({hand.label for hand in hands}) != 1:
        return None, 'Handedness differs between views', None
    points = [canonical(hand) for hand in hands]
    if any(point is None for point in points):
        return None, 'Invalid hand geometry', None
    indices = list(range(5, 21)) if ignore_thumb else list(range(21))
    spread = max(float(np.sqrt(np.mean(np.sum((point[indices]-points[0][indices])**2, axis=1)))) for point in points)
    if diagnostics is not None:
        residual = np.max([np.linalg.norm(point-points[0], axis=1) for point in points], axis=0)
        diagnostics.update(shape_spread=spread, spread_limit=.40,
                           landmark_disagreement=residual.tolist(),
                           finger_disagreement={name: float(np.sqrt(np.mean(residual[start:start+4]**2)))
                                                for name, start in zip(('thumb','index','middle','ring','little'), (1,5,9,13,17))},
                           handedness_agrees=len({hand.label for hand in hands}) == 1,
                           bound_pair=bound_pair)
    if spread > .40:
        return None, 'Views disagree on hand shape', spread
    world = np.mean(points, axis=0)
    fused = Hand(hands[0].pixels, world, hands[0].label, classify(world))
    return fused, 'Combined landmark geometry' if len(hands) > 1 else 'Hand geometry', spread
