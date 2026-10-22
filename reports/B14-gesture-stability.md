# B14: gesture stability under landmark jitter

Classifier version 2 (B2) separates the recorded labelled poses well, but each
frame is judged independently and the engine restarts a gesture hold on any
change. With per-landmark jitter, the thumb IP angle in particular crosses its
140/145-degree gates frame to frame. At 4 mm jitter on synthetic relaxed hands,
82–100% of misclassified frames had the thumb wrong or uncertain. A click needs
about six consecutive identical frames (0.18 s at 30 FPS), and in retained-owner
mode a single missing pair paused input and latched clicks.

## Changes

- **Per-pair hysteresis** (`PoseStabilizer`, applied in `HandOwner` to each
  matched pair). The B2 thresholds are unchanged for a fresh hand. Afterwards an
  uncertain digit keeps its last known state, and a tucked thumb needs 5 degrees
  beyond the open gate to open. Closing uses the normal gate, so a spurious open
  estimate cannot hold a click pose. The per-frame result is kept in pair
  diagnostics as `frame_gesture`.
- **Dropout grace** (0.10 s). Unrecognized or missing frames within the grace
  neither restart the current pose nor count toward its hold, and no action fires
  on a bridged frame. A different recognized pose always restarts the hold. Longer
  gaps keep the existing pause behaviour. Candidates expire only after a real
  interruption, so low frame rates (B9) still complete holds.
- **Stereo check of used joints.** `observe` reprojects the wrist, index tip and
  pitch joints, plus the thumb outside scrolling as click evidence; occluded
  curled ring/little estimates no longer reject the pair.
- **Pointer.** One-Euro filter in place of the 45 ms low-pass; the pointer holds
  still in click poses and a click lands where it was 0.10 s before the pose began.

## Rejected variants

Measured with the same harness (`posed` hands from `tests/test_tracking.py`):

| Variant | Effect |
| --- | --- |
| Resume a pose's hold after a brief different pose | Flickering click poses accumulated hold time; extra clicks at 5 mm |
| Count bridged missing frames toward the hold | About one false right click per 33 min holding two fingers at 6 mm |
| Symmetric thumb margin (sticky open) | Best click success, but false right clicks holding two fingers |
| Distance-based thumb measure | Not adopted: B2's recorded data shows thumb spacing overlaps between open and tucked |

## Limits

These are synthetic results, not real-hand accuracy. Jitter is independent
Gaussian noise per landmark; real model errors are correlated and view-dependent.
The B2 recorded session was not available for replay. The 5-degree margin and
0.10 s grace should be checked against labelled recordings and a live session.
