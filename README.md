# RealMagicMouse

A planned Windows-first pointer controller using a fixed pair of cameras and
hand gestures. macOS support is secondary where dependencies permit it.

## Current status

Requirements and numbered development stages are recorded. There is no runnable
application yet. Camera specifications, calibration workflow, activation meaning,
pointer mapping, and GitHub repository settings are awaiting user input.

See [FEATURES.md](FEATURES.md) for feature status and
[DEVELOPMENT.md](DEVELOPMENT.md) for the incremental build and validation plan.

## Intended interaction

1. Calibrate at startup.
2. Raise a closed fist to activate, hear a subtle chime, and center the cursor.
3. Extend the index finger to move the pointer relative to the activation origin.
4. Extend the thumb with the index finger to left click.
5. Extend the middle finger with the index finger and move to scroll.
6. Extend both thumb and middle finger with the index finger to right click.
7. Open the whole palm to deactivate with a different chime.

Debug mode must show one camera feed, a hand bounding box, and state colors. It
must also work with only one camera; that mode cannot provide measured stereo
depth.

The requested range is up to 12 feet with varied hand orientations. This is a
hardware validation target, not a current capability or guaranteed property.
