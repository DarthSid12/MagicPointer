# MagicPointer — feature 3 checkpoint

Windows-first pointer prototype. This checkpoint implements features 1–3.
Commands available: download-model, targets, cameras, calibrate, run --debug. Later gesture behaviors are not implemented yet.
Hardware validation, native Windows input validation and the 12-foot target remain
pending. See FEATURES.md, DEVELOPMENT.md and VALIDATION.md.

Use 64-bit Python 3.11 on Windows. Create a virtual environment, then run:

```sh
python -m pip install -e .
python -m magicpointer --help
python -m magicpointer download-model
```

Generate and measure printed targets, then calibrate your fixed two-camera rig:

```sh
python -m magicpointer targets
python -m magicpointer calibrate --left 0 --right 1 --fps 30 --square-mm 25 --marker-size 0.20 --marker-spacing 1.00
```

The sizes above are examples: supply measured print sizes and marker-center
spacing. Print at 100%. Capture at least 15 varied checkerboard views with Space,
press C to solve, then show wall markers 0 and 1 to both cameras and Space to save.

For a one-camera preview without OS input: `python -m magicpointer run --debug --left 0 --fps 30`. Q/Escape closes it.
