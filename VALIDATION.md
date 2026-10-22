# Validation record

Validation date: October 20, 2026 (user timezone: America/Los_Angeles).

## Completed locally

- Host: Apple Silicon macOS, Python 3.9.6, isolated .venv. The recommended Windows
  runtime and CI target are 64-bit Python 3.11.
- The combined automated suite ran 33 tests: **32 passed, 1 skipped**. The skipped
  test inspects the native Windows SendInput ABI and runs only on Windows.
- Synthetic stereo reconstruction passed at multiple depths and room positions,
  including the requested 3.6576 m target depth. This verifies mathematics, not
  real-world landmark accuracy or camera resolution at that distance.
- Synthetic rotated/distorted camera geometry, invalid pairs, stereo solve from
  18 varied synthetic checkerboard poses, actual ArUco detection on generated
  images, wall-marker displacement and missing markers were checked.
- All six intended gesture shapes passed synthetic 3D rotation/scale checks.
  These checks do not establish recognition accuracy on real hands.
- Timing checks covered activation, centering, click chord arbitration, click
  latching, scroll entry/direction, palm stop, tracking loss, stalls and rearming.
- A synthetic single-camera application session exercised the actual rendering,
  gesture engine, state changes, chime dispatch and shutdown with OS input disabled.
  A rendered 960 by 540 preview was visually inspected for clipping and legibility.
- Mocked camera-open failures, second-device failure and disconnect paths released
  resources. CLI startup, invalid options, printable target dimensions, package
  install/entry point, byte compilation, dependency checks and whitespace checks
  were validated.
- The official checksum-verified hand model loaded with MediaPipe 0.10.32 and
  processed a blank frame successfully on this Mac. The sandbox blocked macOS
  graphics initialization, so that smoke test ran outside the sandbox without
  opening cameras or sending pointer input.
- MediaPipe 0.10.21 was replaced before delivery after newer pip detected its
  incorrect macOS wheel architecture metadata. Dependency checks pass with the
  selected 0.10.32 build. Native input avoids the initially considered mouse
  library's conflicting macOS build dependencies.

## Completed in GitHub Actions

[Validation run 39482390199](https://github.com/Himir-Desai/MagicPointer/actions/runs/39482390199)
passed on implementation commit `0758819`:

- **Windows / Python 3.11:** all 33 tests passed, including the native SendInput
  structure ABI check. Package installation, dependency checks, CLI startup,
  model download and blank-frame model inference passed.
- **macOS / Python 3.11:** 32 tests passed, with the Windows-only ABI test skipped.
  Package installation, dependency checks, CLI startup, model download and
  blank-frame model inference passed.

These hosted checks use synthetic frames and do not send real pointer events.

## Not yet validated

- Physical cameras: device enumeration, 1080p bandwidth, achievable 30/60 FPS,
  paired-frame timing, lens focus, exposure, real calibration and rig stability.
- Real-hand classification, camera association, varied user positions and hand
  orientations, occlusion, false clicks, pointer latency/jitter and scroll tuning.
- Real 12-foot recognition and pointing accuracy. The code is an initial prototype
  until these measurements are made with the actual rig.
- Native Windows cursor/button/wheel injection, DPI scaling, chimes and camera
  backend; macOS Accessibility input and actual sound playback. Native input APIs
  were implemented but no OS pointer events were sent during local validation.

## Hardware acceptance session

1. Install on the Windows target and run the one-camera debug preview. Confirm
   each gesture, state color, bounding box and orderly exit without desktop input.
2. Generate/measure targets, mount both cameras, collect varied checkerboard
   views, and verify marker size and spacing. Record baseline and RMS errors.
3. Run stereo debug. Check that bad resolution, swapped devices, moved markers,
   missing hands and unmatched pairs prevent control. Confirm fist activation
   works at several positions in the shared view.
4. Test real control on the primary display: center, edges, fine movements,
   single left/right clicks, scroll, palm stop, reactivation and camera disconnect.
5. Repeat at near, middle and 12-foot distances and varied yaw/pitch/roll. Record
   false/missed gestures, jitter, latency, processing FPS and occlusion behavior.
   Adjust thresholds only against measured results; do not label range/orientation
   support verified until this session passes.

## B1: macOS camera authorization guidance

The reported first-use camera authorization failure now provides Allow/rerun,
Camera privacy settings and launcher restart instructions. A mocked macOS failure
regression verifies this guidance and resource cleanup. Local regression run:
34 tests, 33 passed, 1 Windows-only check skipped. No system permissions were
changed; actual permission grant and successful capture require user action.

## B2: Robust gesture recognition

Synthetic anatomical hands with relaxed poses (slightly bent pointing fingers,
loosely curled others) and Gaussian per-landmark jitter were compared against
the previous classifier and engine. Each session holds fist, point, left, point,
right, point, then palm, and succeeds with exactly one activation, one left and
one right click, and one palm deactivation.

| Landmark jitter | Previous (mono / stereo) | Revised (mono / stereo) |
| --- | --- | --- |
| 2 mm | 0% / 0% | 100% / 100% |
| 4 mm | 0% / 0% | 100% / 100% |
| 6 mm | 0% / 0% | 100% / 100% |

The previous pipeline completed the session only without jitter. Per-frame
recognition of relaxed poses at 4 mm rose from 0.1–61% to 93–100% for single
frames and 100% with temporal smoothing. Local regression run (Python 3.11):
44 tests, 43 passed, 1 Windows-only check skipped. Thresholds come from hand
geometry and need tuning against recorded real-hand sessions on the rig.
