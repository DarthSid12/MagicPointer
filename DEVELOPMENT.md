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

## Decisions required before dependent implementation

- GitHub owner, repository name, and visibility. Local Git initialization and
  planning can proceed independently; remote creation cannot.
- Camera models, resolution/FPS, baseline and mounting arrangement. Calibration
  must estimate camera intrinsics, lens distortion and relative pose; knowing the
  baseline alone does not supply these quantities.
- Whether a printed checkerboard visible to both cameras is acceptable for
  calibration. Changing the mount or camera capture geometry invalidates the rig
  calibration. Startup user/workspace calibration is a separate concern.
- Whether cursor mapping uses hand position in a screen-facing interaction plane
  or the pointing direction of the index finger. These need different calibration
  and geometry and have different behavior when the wrist rotates.
- Whether raising a fist means entering a calibrated activation region or
  recognizing an upward arm movement. A hand-only detector does not establish
  arm position.

## Proposed technical direction (subject to feasibility checks)

Python with OpenCV for camera capture/calibration and MediaPipe Hand Landmarker
for hand landmarks is a candidate. Verify Windows package compatibility and pin
versions before selecting the runtime and installing dependencies. Do not treat
model-relative hand coordinates as calibrated stereo depth.

Use measured paired camera observations to reconstruct points in a fixed rig
coordinate system. Associate the same hand across views and reject inconsistent
or stale pairs. Gesture classification should use relative joint geometry rather
than image-axis finger-height comparisons to support varied orientations.

Use a deterministic gesture state engine with temporal stabilization. Full palm
must deactivate before other actions. A stable right-click chord must be resolved
before committing a thumb-only left click or two-finger scroll. The exact timing
thresholds need hands-on tuning and should be documented when implemented.

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
