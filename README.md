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
