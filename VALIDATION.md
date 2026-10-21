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

[B5 CI run 39492043757](https://github.com/Himir-Desai/MagicPointer/actions/runs/39492043757)
passed on `91c798b` for Windows and macOS with Python 3.11: installation,
dependencies, regression tests, CLI startup and model inference. Live wrist
tracking and intended-direction accuracy remain to be confirmed by the user.

## B5 live confirmation

The user reports that wrist-tilt-20261020T203236Z-0dbdfdbb.zip looks good. Replay
exactly reproduces recorded counts, speeds, rejection totals and both neutral
captures. The 815 live-mode frames include 799 valid frames (98.04%) and 16
finger-chain rejections. Outputs include 275 down, 288 up and 236 stop frames;
both directions reach 12 lines/sec. Longest continuous down/up commands are
7.43/6.84 seconds. No thumb-related rejection remains. See
reports/B5-live-confirmation.md and reports/B5-confirmation-replay.json.

This validates the reported fix for one user/session, not general gesture
accuracy or stereo/12-foot/OS-scroll behavior. Runtime code and thresholds were
unchanged; no new unit or UI tests were run for this documentation-only review.

## Feature 7: integrated wrist scrolling; B6 thumb investigation

`run` now replaces translation scrolling with automatic-neutral wrist-rate
scrolling, using the experiment's shared 10-degree dead zone, 30-degree full-speed
angle and bounded 12-step/sec response. Stereo uses triangulated index/middle
direction after both model shapes pass; mono debug uses estimated pitch. Scroll
mode freezes the pointer, ignores thumb labels and suppresses clicks. Index-only
or fist exits; pointer mapping rebases on resume. Valid straight-pair pitch takes
priority over fist-like labels during tilt. Palm stop, timing/depth/hand-identity
checks, tracking-loss deactivation and normal clicks outside scroll remain.

Final local checks: 74 tests run, 73 passed, one Windows-only ABI check skipped.
Coverage includes automatic neutral, dead zone/rate cap, up/down wheel sign,
30/60-Hz accumulation, thumb-label changes, invalid-pitch pause/recalibration,
mode exit without pointer jump, click rearming, palm/tracking-loss stop, stereo
pitch from reconstructed geometry, invalid depth/hand mismatch, and the main
preview/render/dispatch loop without desktop input. Dependency check, CLI startup
and diff whitespace checks passed. Main scroll preview was rendered and visually
inspected. The real model blank-frame smoke initially could not create a graphics
context inside the sandbox; rerunning outside it passed without opening a camera.

Latest user archive through the main engine after seeded activation/scroll entry:
1,048 frames, 90 up wheel steps, 101 down, zero click events and zero pointer
moves; one later tracking-loss deactivation. This is a replay of unlabelled model
landmarks, not intended-direction accuracy. Experiment Space/reset events are not
main-app controls and are ignored after seeding. It does not validate actual mode
entry, stereo camera noise or OS/application scroll distance. Synthetic tests
cover entry and exit; physical stereo and live automatic-neutral checks remain.
See reports/7-scroll-replay.json and reports/B6-thumb-estimates.md. B6 remains open;
no ordinary click-classifier threshold has been changed.

[Feature 7 CI run 39494816348](https://github.com/Himir-Desai/MagicPointer/actions/runs/39494816348)
passed on `fe2a60c` for Windows and macOS with Python 3.11: dependency installation,
regression suite, CLI startup, official model download and blank-frame inference.
Physical stereo cameras and live OS scrolling were not exercised by CI.


## Features 6/7 clarification: B7 entry timing and fresh neutral

Read-only deterministic engine probes with 20-ms samples confirmed one right-click
for simultaneous index/middle/thumb recognition and for an 80-ms two-finger
intermediate. A 240-ms two-finger intermediate latched scrolling first and emitted
no click when the thumb chord followed. This is B7; no runtime fix is included.
Separate entries at +18 and -12 degrees captured +18 and -12 as their respective
neutrals, confirming reset on every entry. Neutral uses at least 300 ms of stable
samples after mode entry, rather than a persisted angle or the first frame alone.
No camera/OS input or full regression suite was run for this documentation-only
clarification; the probes used the unchanged, previously validated engine.

## Feature 11: automatic main-debug diagnostic recording

All valid `run --debug` sessions now create a unique local debug directory and
record continuous samples for mono or stereo preview. Existing tilt experiment
recordings and labelled collection intervals retain their formats. The common
diagnostic-data component is implemented independently of the pending user choice
about additionally saving camera video (subsequently resolved as data only). No
image/video/audio capture was added.

Local checks: 80 tests run, 79 passed, one Windows-only ABI test skipped. New
coverage checks full-session recording including no-hand frames, actual simulated
scroll events and neutral changes, both-camera data and rejection reasons,
nonfinite landmark serialization, flushed samples before close, idempotent
archiving, Ctrl+C/capture/model errors, resource cleanup when archiving fails,
and no recorder for normal desktop-control runs. An existing camera-failure
fixture was updated to supply realistic camera metadata. CLI help and whitespace
checks passed. A synthetic main-preview frame was rendered and visually inspected
for the REC indicator. No live camera capture, OS input or disk-throughput/long-run
performance measurements were performed. Data is flushed per frame; a hard kill
may leave the session folder without its final ZIP.

[Feature 11 CI run 39495901172](https://github.com/Himir-Desai/MagicPointer/actions/runs/39495901172)
passed on `9d8547b` for Windows and macOS with Python 3.11, including installation,
dependency checks, regression tests, CLI startup, model download and blank-frame
inference. Live recording performance and optional camera-video recording remain
outside these checks. The user subsequently confirmed data-only recording; no
video is requested and no runtime change is needed.

Feature 11 clarification: data-only preference confirmed. Documentation and ledger
updated; whitespace checks passed. No tests rerun for this documentation-only
confirmation; runtime remains the CI-validated implementation above.

## Feature 12: two-camera debug preview and calibrated stereo simulation

Two-camera debug now renders both raw feeds side by side with per-view hand
boxes/skeletons, camera indices, handedness/pose/finger estimates, pair grab skew
and calibrated wrist range/virtual pointer when available. Missing, invalid,
wrong-index or wrong-resolution calibration leaves a labelled preview without
joint pointer simulation. Valid calibrated mode retains wall-marker, timing,
hand agreement, depth and tracking-loss checks. Normal desktop control still
requires calibration. Both preview modes record only diagnostic data.

Local final checks: 86 tests run, 85 passed, one Windows-only ABI check skipped.
New end-to-end synthetic coverage exercises both feeds and trackers, all four
calibration-preview fallbacks, marker/hand/timing rejection with visible feeds,
calibrated metric activation and pointer simulation, two-stream archive contents,
partial capture-error archiving and cleanup, and strict real-control calibration.
Existing mono debug, scroll and recording regressions also passed. CLI help and
diff whitespace checks passed. Synthetic pre-calibration and calibrated panels
were rendered and visually inspected for legibility, layout and virtual pointer.

A live macOS smoke test opened camera indices 0 and 1 simultaneously at negotiated
1920x1080 and approximately 30 FPS, initialized both real trackers, displayed 18
pairs, saved a data-only archive and closed successfully. Desktop construction was
explicitly blocked in the test. The archive was checked for exactly the three JSON
files, 18 two-camera samples, no accepted observations/events in uncalibrated mode,
no image recording and normal termination. The local archive is
`diagnostics/debug-20261021T230215Z-300b51ff.zip` (not committed). A separate probe
found usable indices 0-3; these are device indices, not verified physical model
identities. The user should select the intended pair by viewing their feeds.

There is no calibration.json for this rig, so live calibrated depth/gesture
accuracy, wall-marker acceptance, long-run throughput and Windows hardware have
not been established by this prompt. Calibrated mode is covered with synthetic
stereo geometry and mocked marker results; actual marker geometry retains its
existing unit coverage. No camera images or video were saved.

[Feature 12 CI run 39690791082](https://github.com/Himir-Desai/MagicPointer/actions/runs/39690791082)
passed on `06fc40c` for Windows and macOS with Python 3.11: dependency installation,
regression suite, CLI startup, model download and blank-frame inference. The live
macOS preview test above additionally exercises real simultaneous capture;
calibrated stereo hardware accuracy remains unverified.

## 3.1 / 4 revamp / 12.1 / B8: readable shared gestures and one-hand ownership

Main debug now displays a large shared gesture label and a separate fist-countdown,
locked-hand or paused status banner. Main capture detects up to four hands per view
(configurable 1-8); local spatial tracking preserves IDs independently of model
list order. One continuous matched fist held for 2.5 seconds selects the owner and
emits the activation chime. Input uses only that owner until its palm releases it.
Tracking loss pauses/deactivates input but retains ownership; other hands do not
get to activate or release an existing lock. Ambiguous crossings/pairs pause.

Both camera views are normalized into a palm coordinate frame, their landmark
geometry is averaged, and the gesture is classified once. Raw per-view gesture
labels remain in the small diagnostics and recording; they are not independent
input decisions. Both views must support an unambiguous match, with no one-view
fallback (user-confirmed). Handedness, shape consistency and temporal continuity
support association before calibration; calibrated mode also validates stereo
anchors. These are heuristic tracking/association checks, not biometric identity
or a trained multi-view gesture network. Similar same-handed hands can require
separation before a match becomes unambiguous.

Uncalibrated debug now permits shared-gesture activation and simulated pointer,
click and wrist-scroll events without printouts. Position is primary-view image
space normalized by activation palm size; this does not estimate metric depth.
`--uncalibrated` explicitly forces testing even with an existing calibration file.
The flag cannot enable real OS input. Normal desktop control retains calibrated
stereo geometry and wall-marker checks. Session archives now include every detected
hand, track IDs, candidate/owner IDs, selection progress and combined geometry.

Validation checkpoints:

- Tracking/fusion foundation staged tree: tested in an isolated snapshot of the
  previous commit plus staged component changes, 96 tests run, 95 passed, one
  Windows ABI skip. This confirms that rollback checkpoint still runs with the
  original application/fixtures. No intermediate integration files were included.
- Final integration: 100 tests run, 99 passed, one Windows ABI skip. Coverage adds
  view rotation/scale/translation normalization, shared classification independent
  of raw pose labels, required two-view evidence, model multi-hand extraction and
  legacy single-hand compatibility, ID order changes, ambiguous crossings, unique
  fist selection, palm release, selection after loss, and uncalibrated bypass guards.
- An end-to-end two-camera scene switches the fist between hands before the dwell,
  locks only after a continuous 2.5 seconds, ignores the other hand's palm, pauses
  when the owner's second view disappears, resumes the same IDs, releases on the
  owner's palm and then selects the other hand. No click originates from distractors.
- Countdown and locked/shared-point panels were rendered from synthetic data and
  visually inspected. The main banner is substantially larger than per-view text;
  paused state and no-metric-depth labels remain separate. CLI help and whitespace
  checks passed. CI model smoke now initializes the four-hand model configuration.
- Live macOS capture/model/render/archive smoke: cameras 0 and 1, 1920x1080, about
  30 negotiated FPS, two four-hand trackers, 24 displayed pairs, normal exit and
  data-only archive. Desktop construction was explicitly prohibited. The archive
  contains both-camera hand lists and selection diagnostics, but no hands were
  detected during this short test. It does not establish live gesture accuracy,
  owner recovery under occlusion, audible chime acceptance or long-run latency.

The live local archive is `diagnostics/debug-20261021T233028Z-b47554b5.zip` and is
not committed. B8 remains open for actual multi-hand confirmation. B6 thumb-estimate
accuracy and B7 click/scroll entry timing were not independently changed by this
work. Calibrated hardware/12-foot acceptance remains pending.

[Ownership/fusion CI run 39694283068](https://github.com/Himir-Desai/MagicPointer/actions/runs/39694283068)
passed on integration commit `98f6295` for Windows and macOS with Python 3.11:
installation, dependency checks, complete regression suite, CLI startup, official
model download and real four-hand model blank-frame inference. The watcher briefly
lost its network connection; the final run/job API result independently confirmed
success for both jobs. No test rerun or code correction was needed. Live hand
selection and fusion accuracy remain pending user testing; B8 stays open for that
confirmation. Rollback foundation commit `60c9be8` was separately snapshot-tested.
