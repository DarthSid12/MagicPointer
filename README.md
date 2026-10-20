# MagicPointer

Windows-first stereo hand pointer, with an optional macOS input adapter and a
single-camera debug preview. The implementation is an initial prototype: real
Windows input, camera-rig accuracy, gesture recognition on real hands, and the
12-foot target require hardware validation. See [VALIDATION.md](VALIDATION.md).

## Install

Use 64-bit Python **3.11** on Windows 10/11. The package supports Python 3.9–3.12;
3.11 is the CI target. Open a terminal in this checkout:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m magicpointer download-model
python -m magicpointer cameras --fps auto
```

If PowerShell activation is restricted, run `.venv\Scripts\python.exe` directly
instead of `python`. Model download requires internet access once; normal camera
processing is local. The versioned official MediaPipe model is checksum-verified.

On macOS, use a compatible Python installation:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[macos]'
python -m magicpointer download-model
```

Allow camera access for the terminal/Python. Actual macOS pointer control also
needs Accessibility permission in System Settings, followed by a restart of the
terminal/app. Debug mode does not request desktop input permission. Windows uses
native input APIs, with no macOS-only dependencies in the default installation.

If OpenCV prints `not authorized to capture video (status 0)` on macOS, the
camera permission request has not completed. Choose **Allow** in the macOS
prompt, then rerun the command. If it still fails, open **System Settings →
Privacy & Security → Camera**, enable the app launching Python (such as VS Code,
Terminal, or iTerm), then fully quit and reopen that app. An IDE's integrated
terminal may use the IDE's permission rather than Terminal's. Reopen this project
and retry the debug command below. The program cannot grant itself camera access.
If permission is already enabled, check the device connection and camera index.

The initial Matplotlib font-cache message, GPU/Metal information, XNNPACK CPU
delegate message and feedback-tensor warnings do not explain this permission
failure; the model has initialized before the camera is opened.

## Try one camera first

```sh
python -m magicpointer run --debug --left 0 --fps 30
```

This displays the primary camera, hand box/skeleton, gesture, system state,
processing FPS, and a small virtual screen. It **never moves or clicks the OS
pointer**. Gray is inactive, green active, purple click feedback, blue scrolling,
and orange unavailable tracking. The feed is the raw camera view; horizontal
pointer mapping is mirrored by default for a user facing the TV.

Mono preview estimates scale from palm size at activation. It cannot measure
stereo depth or establish 12-foot accuracy. Only one hand is tracked per view.
Keep other hands out of view while controlling the pointer.

## Prepare and calibrate the fixed stereo rig

Place both 1080p cameras near the TV with overlapping views and a rigid mount.
Use separate camera indices. Do not move either camera, change its lens/zoom, or
change capture geometry after calibration. Generic USB cameras use software-paired
frames; they are not hardware synchronized. Fast hand motion can reduce accuracy.

Generate printable targets:

```sh
python -m magicpointer targets --square-mm 25 --marker-mm 200
```

Print the generated SVGs at **100% scale**, using tiled printing or a larger sheet
where necessary. Measure the print: the checkerboard has 10 × 7 squares (9 × 6
inner corners); `--square-mm` is one square's measured edge. Marker size is the
outer **black square** edge, excluding the surrounding white margin.

Attach ArUco markers **0 and 1** (`DICT_4X4_50`) to the opposite wall so both are
visible in both cameras. Record their black-square size and center-to-center
spacing. The example below uses 20 cm markers separated by 1 m; substitute your
actual measurements:

```sh
python -m magicpointer calibrate --left 0 --right 1 --fps 30 --square-mm 25 --marker-size 0.20 --marker-spacing 1.00
```

Move and tilt the checkerboard through both cameras' shared field of view. Press
**Space** for each of at least 15 distinct views at varied positions, tilts and
depths. Press **C** to solve. After solving, remove the board, show both permanent
wall markers, and press **Space** to verify measured size/spacing and save.
Calibration errors over 2 pixels are rejected. Use Q/Escape or close the window
to cancel without replacing the previous calibration.

The checkerboard establishes lens intrinsics/distortion and stereo pose in metric
units. Permanent markers verify the fixed setup at each startup; two marker
centers alone do not replace camera calibration. Camera-to-wall distances are
inferred from calibrated stereo geometry rather than required as extra inputs.

## Run stereo control

Preview the complete stereo path before using desktop input:

```sh
python -m magicpointer run --debug --left 0 --right 1 --fps 30
python -m magicpointer run --left 0 --right 1 --fps 30
```

Both commands load `calibration.json` and verify the wall markers before allowing
activation. Marker verification replaces repeating the full checkerboard process
when the rig stays fixed. During use, visible markers are checked periodically;
a detected displacement stops the app and requires recalibration. Hidden markers
cannot reveal a moved rig, so keep them visible when possible.

**Q, Escape, Ctrl+C, or closing the window stops the application.** An open palm
also deactivates pointer control while keeping the application running. Camera
failures end the app with a readable error; tracking loss stops input immediately
and deactivates after 0.30 s. A processing stall over that threshold also requires
reactivation.

## Gestures

| Gesture | Result |
| --- | --- |
| Closed fist anywhere in the shared view, held 0.45 s | Activation chime; fist wrist becomes origin; cursor centers on primary display |
| Index extended, other fingers curled | Index tip controls pointer relative to origin |
| Index + thumb | One left click after 0.18 s of stable recognition |
| Index + middle | Vertical two-finger movement scrolls after stabilization |
| Index + thumb + middle | One right click after 0.18 s of stable recognition |
| Entire palm open | Stop input immediately; confirm deactivation after 0.18 s with a different chime |

Open the fingers for a click chord together: holding thumb-only long enough is a
left click, even if you later add the middle finger. After a click, return to
index-only or a fist for 0.18 s before another click or scroll. Holding a click
pose does not repeat clicks. Unrecognized poses produce no pointer actions.

The interaction plane is parallel to the camera image plane, centered at the
activation fist. Both whole-hand motion and index-tip motion affect it. A default
40 × 25 cm workspace maps to the primary display; depth changes do not resize it.
The cursor is driven by tip **position**, rather than a ray projected along the
finger. Ring and little fingers stay curled for pointing/clicking/scrolling.

```sh
python -m magicpointer run --left 0 --right 1 --fps 60 --workspace-width 0.50 --workspace-height 0.30
```

`--fps` accepts `auto`, `30`, or `60`. Auto requests 60, falling back to 30 when
the driver reports a lower rate. Hardware may ignore these requests; the console
reports negotiated resolution/FPS and the window reports actual processing FPS.
Two 1080p cameras can exceed shared USB bandwidth, especially at 60 FPS.

Other options: `--no-mirror-x`, `--mute`, `--model PATH`, `--calibration PATH`.
Run `python -m magicpointer run --help` for details.

## Limits and development

### Try finger-tilt scrolling

```sh
python -m magicpointer tilt --camera 0 --fps 30
```

This one-camera experiment moves a **simulated document** beside the camera feed.
It sends no OS pointer, click or scroll events. The regular `run` scrolling
behavior is unchanged.

1. Support your arm comfortably. Extend index and middle, keep them reasonably
   straight, and curl ring/little fingers with the thumb tucked.
2. Hold your fingers in a comfortable neutral position for half a second, then
   press **Space** to capture neutral.
3. Keeping the palm steady, tilt the two fingers **at their base knuckles** in
   your desired downscroll direction. Hold steady, then press **D**. Aim for
   about 15–30 degrees of comfortable tilt; at least 10 degrees is required with
   the default dead zone. This establishes direction and full-speed range for
   either hand. Whole-wrist rotation alone is intentionally ignored.
4. Return to neutral to stop. Tilt down to scroll continuously down; tilt the
   opposite way for upscroll. Tilt farther for greater speed. The yellow cross
   stays fixed over the simulated document.
5. Try holding neutral, down and up; then translate or rotate the whole hand while
   keeping the fingers fixed relative to the palm. Note unwanted drift, jitter,
   pose loss or an uncomfortable range. **R** resets calibration.

**Q/Escape**, closing the window, or Ctrl+C saves and exits. Normal classifier
results are displayed beside the experimental shape check, which allows base
knuckle tilt while still requiring extended finger chains. Invalid/one-finger/
open-palm poses stop simulated scrolling immediately. Loss of a valid shape for
over 0.30 seconds, a processing stall or a handedness change clears calibration;
capture neutral and down again when prompted.

The default neutral dead zone is 5 degrees (`--dead-zone 7` increases it). Speed
is capped at 12 simulated lines per second with 0.10-second smoothing. Calibration
captures require a steady 0.30-second window with no more than 6 degrees of range.
This is a feasibility experiment, not evidence of tracking accuracy at 12 feet.

The terminal prints a `diagnostics/tilt-...zip` containing palm-relative angles,
landmarks, normal pose classification, simulated speed, calibration key events
and summary counts. No images, video or audio are stored or uploaded. Share this
zip with your observations so tilt reliability can be evaluated before changing
the actual scroll controller.

### Record gesture diagnostics

If real poses become unrecognized when you rotate your hand, use the guided
one-camera collector:

```sh
python -m magicpointer collect --camera 0 --fps 30
```

The window guides you through six gestures at five wrist angles (30 short clips).
Use the same hand and typical room lighting/distance throughout a session. Keep
only that hand in view. For each prompt, make the requested pose **first**, then
press **Space** and hold it for three seconds. The expected pose is your manual
label, independent of what the classifier predicts; keep recording even if it
says unrecognized. Each new prompt waits for Space so transitions are not labelled
as the next gesture. **S** skips a pose; **Q/Escape**, closing the window or Ctrl+C
saves a partial session. Incomplete clips are marked in the summary.

This mode never controls the pointer. It saves no photos, video or audio and
uploads nothing. Files contain hand landmarks, timing, camera settings,
handedness, predicted versus requested poses and the classifier's joint-angle,
extension and thumb measurements. Handedness score is not gesture confidence.
The on-screen state values are 1 extended, 0 curled, -1 uncertain.

The terminal prints a timestamped `.zip` under `diagnostics/` containing
`session.json`, `samples.jsonl`, and `summary.json`. Share that archive in this
chat so the real failures can be measured and the classifier corrected. Different
distances or hands should use separate sessions. `--seconds 5` records longer
clips; `--output PATH` changes the local destination. The default diagnostics
directory is excluded from Git. Recordings can be replayed against the current classifier using
`python scripts/evaluate_gestures.py PATH_TO_ZIP`. Classifier version 2 corrects
the strict joint-angle and thumb-spacing rules identified in the first session;
see [B2 analysis](reports/B2-analysis.md) for measured improvements and remaining
limits. Live confirmation is still needed.

The requested 12-foot range (3.6576 m) is a validation target, not a measured
claim. Real control rejects wrists beyond that measured range. Resolution,
lighting, exposure, camera spacing, occlusion and model quality affect results.
Joint-angle classification is rotation invariant in landmark coordinates, but
real landmark estimation is not guaranteed at every orientation. Fingers hidden
from both cameras cannot be reliably observed.

Tests do not move the real cursor:

```sh
python -m unittest discover -s tests -v
python -m pip check
```

[FEATURES.md](FEATURES.md) tracks the eight numbered features;
[DEVELOPMENT.md](DEVELOPMENT.md) records the design and commit boundaries;
[BUGS.md](BUGS.md) preserves reported bugs. Each implementation stage is intended
to start cleanly with unavailable later features; stage commits share this final
validation phase before being recorded.
