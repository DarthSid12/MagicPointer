# 14: smoothness and reliability audit

Scope: the whole run path, from camera capture to OS input. Gesture
classification and engine timing were covered separately by B15. All
measurements below are local and synthetic; none use the user's cameras.

## Findings

| Area | Finding | Evidence |
| --- | --- | --- |
| Capture | Cameras were read one after another with blocking `grab()`. Each loop waited for each camera's next frame, or processed a frame the driver had buffered earlier. | B9 recorded 7.75 FPS and 10% pair-timing rejects; B14 recorded 14.86 FPS with only ~34 ms of inference per frame. |
| Inference | The two detectors ran sequentially. MediaPipe releases the GIL, so they can overlap. | Two 1080p detections: 32.7 ms sequential, 16.7 ms threaded (this Mac). |
| Windows timers | `cv2.waitKey(1)` and short sleeps round up to the default ~15.6 ms timer tick on every loop. | Windows default timer resolution; modelled below. |
| Marker checks (B16) | ArUco detection on both full frames ran inside the loop once a second, and a single bad reading raised and ended the session. | 5 ms per 1080p image on a clean scene, 122 ms on a noisy one: up to ~250 ms per check, longer than the 0.30 s loss timeout. |
| Camera loss | Any single failed `grab()` or `retrieve()` ended the app. | `cameras.py` before this change. |
| Pointer | The OS pointer moved only once per vision frame, so it stepped visibly at 10–30 Hz. | `desktop.move` per engine event. |
| Scrolling | Scroll was sent as whole wheel lines, a few per second. | `desktop.scroll(int(lines))`. |
| Detection at distance | The palm detector sees each frame shrunk to about 192 px, so a hand a few metres away is only a few pixels wide. | B10 (loss with distance); B14: 44–60% per-view detection availability, median palm 65–80 px at 1080p. |
| Activation | The 2.5 s fist hold was hard-coded. | `app.py`. |

## Changes

1. **Background capture** (`cameras.py`). One thread per camera keeps only the
   newest frame. `read()` returns frames newer than the last ones returned, so
   the loop never waits on a buffered frame or processes one twice. With two
   cameras, a pair skewed beyond the 40 ms tolerance waits up to 60 ms for the
   older view's next frame and re-takes the other view's newest frame. A stream
   with no frames for 0.5 s is reopened in the background; only 5 s without a
   frame ends the app.
2. **Parallel detection** (`app.detect_views`). A two-worker pool runs both
   cameras' detectors at once. Per-camera timings are still logged, plus
   `detect_total_ms`.
3. **Background marker checks** (`app.MarkerWatch`, B16). Startup verification
   is unchanged. Periodic checks run on a worker thread, and three consecutive
   failures are required to stop the app.
4. **Windows timer resolution.** `timeBeginPeriod(1)` while the app runs,
   restored on exit.
5. **Smooth pointer and scrolling** (`desktop.SmoothDesktop`). A 120 Hz thread
   eases the OS pointer toward the latest target with a time constant of half
   the observed frame interval (8–60 ms), and stops making OS calls once the
   pointer has settled. Clicks first land the pointer exactly on target. Scroll
   lines drain as fine increments over about 80 ms: 1/120-line wheel units on
   Windows, pixels on macOS. `--no-smooth-scroll` restores whole lines. An OS
   error on the background thread is re-raised by the next main-loop input call.
6. **Crop recovery** (`recovery.CropRecovery`, `--no-crop-recovery`). When a
   view loses a hand seen in the last 1.5 s, a separate IMAGE-mode detector
   searches a square crop of three times the hand box (at least 256 px) around
   its last position. The normal 0.65 thresholds apply. A found hand is kept
   only near the lost position and if the regular detector has not already
   found it, and it then goes through ownership, fusion and stereo checks as
   usual. With an owner locked, the owner's position is remembered; otherwise
   the largest hand in each view.
7. **`--activation-seconds`** (default 2.5) for the fist hold.

## Measurements

Full `run --debug --uncalibrated` loop, real MediaPipe detection on 1080p
frames, two simulated free-running 30 FPS cameras with a one-frame buffer.
Each image carries its true capture time, so frame age is measured when
detection starts.

| Configuration | Loop rate | Frame age at detection (median / p95) |
| --- | --- | --- |
| Before, this Mac | 27.7 FPS | 25.0 / 46.6 ms |
| After, this Mac | 30.0 FPS (camera-limited) | 11.9 / 23.3 ms |
| Before, Windows timer modelled (15 ms `waitKey`) | 17.1 FPS | 24.5 / 44.0 ms |
| After, 1 ms timer modelled | 30.0 FPS (camera-limited) | 11.8 / 23.3 ms |

The modelled "before" rate is close to B14's recorded 14.86 FPS. Real Windows
capture and inference costs, and the 1 ms timer's effect, still need a live run.

## Not changed

- **Real control without calibration** stays disabled, as designed. The guard is
  at the top of `app.run`.
- **Gesture set and B7** (right-click versus scroll entry) are unchanged.
- **The 40 ms pair tolerance** is unchanged; cameras that drop to 15 FPS in low
  light can still exceed it.
- **Crop recovery** is validated with mocked detectors only. Its effect on
  real distant hands, and any false detections, should be checked in a debug
  archive (`crop_recovered_frames_per_camera`, `crop_recovery_ms`).
