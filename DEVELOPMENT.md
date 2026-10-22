# Incremental development plan

## Delivery rules

Implement features in dependency order. Each completed feature must leave an
application that starts and handles missing prerequisites without an unhandled
exception. Keep unfinished functionality clearly unavailable.

Batch non-instantaneous checks at the end of a development prompt as required by
AGENTS.md. After successful validation, create focused commits for completed
features. Do not describe a planning commit as working pointer software. Record
the validation and commit for each completed stage in the feature ledger.

Windows is the primary platform. Keep tracking and gesture decisions separate
from operating-system pointer injection so core behavior can be tested without
moving the developer's real cursor. Platform support requires actual validation;
running on macOS does not establish that Windows input injection works.

## Feature boundaries and acceptance

| ID | Deliverable | Acceptance checks before completion |
| --- | --- | --- |
| 1 | Runnable application shell, explicit startup errors, configuration, Windows installation instructions | Starts on supported Python/runtime; invalid configuration and missing optional dependencies produce actionable errors; Windows validation recorded separately from host checks. |
| 2 | Camera selection, paired capture, intrinsic/extrinsic calibration and quality checks | Known calibration target; measured geometry; reject invalid calibration, missing cameras, stale or mismatched frame pairs; release cameras on exit. |
| 3 | Hand landmarks and debug feed with hand box and state legend | One camera works without stereo calibration; no hand, occlusion, disconnect and clean exit handled; displayed state agrees with engine state. |
| 4 | Stable closed-fist activation, origin capture, cursor centering, subtle activation chime | Inactive gestures produce no pointer actions; one activation/chime per accepted transition; activation cannot occur without valid tracking and required calibration. |
| 5 | Continuous index-tip mapping and distance compensation | Test center, screen edges, jitter, depth changes and lost tracking; validate chosen mapping against known synthetic geometry and actual camera observations. |
| 6 | Left/right click recognition and OS adapter | One click per held gesture; release required to repeat; right-click chord does not accidentally emit a left click while fingers transition; verify actual Windows button events. |
| 7 | Two-finger scrolling | Scroll direction and rate follow motion; entering scroll does not click; right-click chord takes precedence over scrolling; verify actual Windows scroll events. |
| 8 | Full-palm stop and distinct chime | Deactivation stops movement/click/scroll, emits one chime, clears session origin, and requires another valid fist activation; lost tracking never continues stale actions. |

The physical accuracy checks require the user's rig. Synthetic tests can verify
geometry and event ordering, but cannot establish long-range recognition quality.

## Confirmed decisions

- The user owns the existing remote at
  https://github.com/Himir-Desai/MagicPointer. The checkout belongs to local user
  `himirdesai`; commits use the configured Himir Desai identity. Development is on
  `feat/pointer-runtime`, based on the initial planning commit on `main`.
- Two 1080p cameras; expose auto/30/60 FPS selection and report driver-negotiated
  settings. The physical models, achievable rate and baseline remain to be tested.
- User approved square ArUco wall markers plus a temporarily moved checkerboard.
  Generated targets use ArUco dictionary DICT_4X4_50, IDs 0 and 1, and a checkerboard
  with 9 by 6 inner corners. Printed dimensions and marker spacing are supplied at
  calibration time. Startup verifies the saved rig against the wall markers.
- A closed fist may activate anywhere in the shared camera view. No arm/body
  detector or fixed activation zone is required.
- Position of the index tip, including whole-hand movement, drives a local plane
  parallel to the camera image plane. Its origin is the activation fist wrist.
  The default physical workspace is 0.40 by 0.25 m and may be changed at startup.

## Implemented technical direction

Python with OpenCV for camera capture/calibration and MediaPipe Hand Landmarker
for hand landmarks. Direct dependencies are pinned in pyproject.toml. Windows
uses native Win32 input; macOS Quartz support is an optional extra. Model-relative
hand coordinates classify gestures; calibrated paired observations supply metric
depth. Physical-platform validation is still required.

Use measured paired camera observations to reconstruct points in a fixed rig
coordinate system. Associate the same hand across views and reject inconsistent
or stale pairs. Gesture classification should use relative joint geometry rather
than image-axis finger-height comparisons to support varied orientations.

Use a deterministic gesture state engine with temporal stabilization. Full palm
must deactivate before other actions. A stable right-click chord must be resolved
before committing a thumb-only left click or two-finger scroll. The exact timing
thresholds need hands-on tuning and should be documented when implemented.

Implemented initial timings: fist activation 0.45 s, gesture confirmation 0.18 s,
tracking loss 0.30 s, dropout grace 0.10 s, click position lookback 0.10 s.
Each digit is classified from one long-span extension ratio with hysteresis;
stereo fuses both cameras' ratios rather than requiring identical per-camera
gestures. The pointer uses a One-Euro filter (1 Hz minimum cutoff, beta 10). Clicks latch until a stable index-only or fist pose; a right
click cannot fall through into scrolling as its thumb closes. Chord fingers must
be opened within the stabilization interval to avoid being accepted separately.
Unknown/missing observations produce no OS input. Control is limited to the
primary display; debug mode always uses a virtual cursor.

## Rollback checkpoints

The implementation is committed in feature order, with a runnable CLI at the
foundation checkpoint, calibration tools at checkpoint 2, a camera preview at
checkpoint 3, then activation, movement, clicks, scrolling and palm deactivation.
Earlier checkpoints expose only commands implemented at that checkpoint. Later
gesture behaviors are added to the engine in their own commits. The final phase
validates each staged checkpoint before its commit and the combined application.
Hardware validation is explicitly pending rather than represented as passed.

## Range and orientation validation

Evaluate the final rig at near, middle and 12-foot distances, recording camera
settings, baseline, lighting, frame rate, hand size in pixels, pointer jitter,
latency, missed gestures, false clicks and tracking loss. Repeat with both hands
and varied yaw, pitch and roll. Report observed limits instead of claiming
unrestricted orientation support. Fingers hidden from both views cannot be
reliably observed by stereo reconstruction.

## Primary references

- [OpenCV camera calibration and 3D reconstruction](https://docs.opencv.org/4.4.0/d9/d0c/group__calib3d.html)
- [MediaPipe Hand Landmarker for Python](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python)

These references support the candidate APIs, not a guarantee of the requested
12-foot operating range.
