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

## Feature 9 / B2: guided real-hand diagnostics

Local validation: 39 tests, 38 passed, 1 Windows-only ABI check skipped. Tests
cover all six gestures/five orientations, raw measurement/classifier consistency,
missing-hand denominators, summaries, unique sessions, archive contents and
partial recording on quit. A mocked live collection exercised rendering and
cleanup without constructing desktop input; the prompt rendering was visually
inspected. CLI help and whitespace checks passed. Recognition thresholds and
pose decisions are unchanged; their metrics are now exported by the classifier.
No user hand recordings have yet been collected. B2 remains unfixed pending
labelled samples and a measured recognition correction.

## B2 correction from supplied landmark recording

The supplied archive was replayed locally without OS input. Recognition matches
improved from 373/2,592 (14.4%) to 2,553/2,592 (98.5%). All 30 labelled clips
reach the engine confirmation dwell. Left/right click, scroll and full-palm
recordings match in every frame. The 39 remaining uncertain frames are confined
to right-rolled pointing. See reports/B2-analysis.md and reports/B2-replay.json
for reproducible aggregate results and limitations. The session informed the
correction; these are not independent test accuracy figures.

Local regression suite: 43 tests, 42 passed, one Windows-only ABI check skipped.
No cameras or pointer events were used. Real-time replay, natural transitions,
false click rates, stereo control and other users still require live validation.
B2 remains open for live confirmation and remaining pointing uncertainty.

[B2 CI run 39486515772](https://github.com/Himir-Desai/MagicPointer/actions/runs/39486515772)
passed on commit `c59a8cf` for Windows and macOS with Python 3.11: package install,
dependency checks, regression suite, CLI help, model download and blank-frame
inference. These hosted checks do not establish live recognition accuracy.

## Feature 10: palm-relative finger-tilt experiment

Local regression suite: 52 tests, 51 passed, one Windows-only ABI check skipped.
New checks cover geometric invariance under whole-hand translation/rotation/scale,
calibration in both direction conventions, neutral dead zone, signed capped
speed, invalid/missing/one-finger/palm poses, tracking loss, processing stalls,
handedness changes, rejected calibration windows, monotonic timestamps and invalid
configuration. A synthetic UI session verified no desktop construction, archive
contents, resource cleanup and partial-session preservation after camera failure.
The 1200-by-720 simulated page and camera preview were rendered and visually
inspected for legibility and clipping; CLI help and whitespace checks passed.

No physical camera capture or OS pointer/scroll events were used. Calibration,
comfort, model angle jitter and usability need the user's live experiment. The
real scroll controller and pointer mapping were not modified. Tilt logs contain
landmarks, angles and calibration events locally under ignored diagnostics/.

[Feature 10 CI run 39488485036](https://github.com/Himir-Desai/MagicPointer/actions/runs/39488485036)
passed on `2da63b7` for Windows and macOS with Python 3.11, including installation,
dependency checks, tests, CLI startup and model inference. Physical-camera tilt
comfort and reliability remain to be tested by the user.

## Experiment 10.1: camera-relative wrist tilt

Local regression suite: 59 tests, 58 passed, one Windows-only ABI check skipped.
Tests cover whole-hand pitch response, yaw/translation/scale behavior, neutral
roll, captured neutral with a symmetric 10-degree dead zone, up/down speed and
capping, sideways/away/folded rejection, tracking loss, reset and invalid full-speed
angles. A mocked wrist UI session verified separate archive kind/filename,
metadata, no desktop construction and camera/tracker cleanup. Existing finger
tilt tests continue to pass. Wrist CLI help and whitespace checks passed, and a
1200-by-720 rendered preview was visually inspected for clipping and legibility.

The command is `wrist-tilt`; the old `tilt` experiment and real `run` behavior
are preserved. One steady neutral capture starts simulated scrolling, with no D
calibration. The direction estimate uses model camera-frame landmarks rather
than calibrated stereo geometry. No real camera, pointer events or near-lens
reliability tests were run; the user must evaluate foreshortening, comfort and
noise using the local wrist-tilt diagnostic archive.

[Experiment 10.1 CI run 39490932959](https://github.com/Himir-Desai/MagicPointer/actions/runs/39490932959)
passed on `c63f7aa` for Windows and macOS with Python 3.11: installation,
dependencies, regression tests, CLI startup and model inference. This does not
verify physical-camera wrist direction or foreshortened hand tracking.

## B5: wrist experiment thumb exclusion

Local regression suite: 62 tests, 61 passed, one Windows-only ABI check skipped.
The supplied 1,952-frame wrist archive was replayed through the updated controller
with its original neutral/reset key events. Accepted frames increased 586 → 938,
down command frames 2 → 96, maximum down output 2.09 → 12 lines/sec. These are
simulated output counts, not labelled intent accuracy; the recording informed
the correction. No raw recordings were committed or OS input emitted.

Synthetic tests cover open-thumb downscroll, unchanged ordinary gesture/finger
experiment semantics, retained fist/full-palm rejection and offline replay.
Remaining no-hand, folded-finger and direction failures are still rejected. See
reports/B5-wrist-analysis.md and reports/B5-wrist-replay.json for the source hash,
calibration events, exact aggregates and live-test limitations.
