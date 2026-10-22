# B14 / B10: Windows left/right detection comparison

Source: `debug-20261022T010423Z-856cc2a1.zip`, supplied by the user.
SHA-256: `b8ca65cfbc1d76d499bab53a54114c0ce8ae5281dc8096dd2b4a9236c36400f8`.
Archive revision: `8faceeb`; clean Windows checkout, Python 3.12.10,
MediaPipe 0.10.32, classifier 2, shared palm-aligned fusion 1.
This report analyzes data only. No runtime settings or camera policies changed.

## Finding

The right view has substantially lower hand-detection availability. The log does
not establish that its gesture labels are less accurate: it contains no intended
gesture labels or images. Once a hand is detected, both views assign a named
gesture to approximately 82% of detections. The left view also has many missing
detections; “always recognized” is not supported for the whole session.

There are 4,215 sampled frame pairs. Session elapsed time ends at 294.531 seconds;
the first sample is at 10.921 seconds, so sampled coverage spans 283.610 seconds.
Processing averages 14.86 frame pairs/sec. Both cameras report MSMF capture at
1920 x 1080 and 30 FPS. Thus this recording is actually 1080p on both sides,
regardless of any lower requested size. Stereo calibration is explicitly off.

| Measure | Left / primary / index 1 | Right / secondary / index 2 |
| --- | ---: | ---: |
| Samples with at least one raw hand detection | 2,527 / 4,215 (59.95%) | 1,840 / 4,215 (43.65%) |
| Samples with at least one named gesture estimate | 2,138 / 4,215 (50.72%) | 1,534 / 4,215 (36.39%) |
| Raw hand detections, including multiple hands in a sample | 2,632 | 1,888 |
| Named gesture estimates among raw detections | 2,154 / 2,632 (81.84%) | 1,544 / 1,888 (81.78%) |
| Unrecognized estimates among raw detections | 478 | 344 |
| Median wrist-to-middle-knuckle distance, detected hands only | 64.9 px | 79.7 px |

“Named gesture” means any classifier result other than `unrecognized`, not a
verified correct gesture. Samples without hands may include periods when the user
intentionally had no hand in view; these percentages are availability, not recall.

| Raw detection availability | Samples |
| --- | ---: |
| Both views | 1,722 |
| Left only | 805 |
| Right only | 118 |
| Neither | 1,570 |

The imbalance is not constant. Between elapsed 60–90 seconds, right detection
occurs in 363/421 samples versus left 341/421. Between 210–240 seconds, left has
detections in 343/467 samples and right in only 24/467. Visibility, posture, focus
or lighting might vary over those intervals; this archive cannot identify which.

In the 1,208 samples with a bound owner, raw detections are absent in the left
view in 67 samples and right in 139. Another 12 left / 2 right samples contain
other detections but no matching owner ID. This supports investigating the right
view's detector availability before attributing all losses to identity matching.
Pair timing rejects are zero, but host grab timestamps do not prove synchronized
exposures or exclude stale/buffered frames.

For 1,616 samples with exactly one detected hand in each view, left has 212
unrecognized estimates (13.12%) versus right 254 (15.72%). There are also label
differences such as left `point` / right `left click` in 83 samples. Without
intended-pose labels, neither view can be declared correct. The right palm is
not generally smaller in this common subset either: median 79.9 px versus left
73.1 px. All size statistics exclude missing detections and cannot measure hand
size or actual distance during failures.

## Code asymmetry review

Inspected the archived revision's capture, tracker, main-loop, ownership and
fusion paths. The detection/fusion files match current main at analysis time.

- Both feeds use MSMF and the same requested capture settings and format policy.
- Each gets its own CPU HandLandmarker, using the same model, VIDEO mode,
  four-hand limit and 0.65 detection/presence/tracking thresholds.
- Both raw images receive the same BGR-to-RGB conversion, without a side-specific
  crop, resize, mirror or calibration correction before inference.
- Both are processed once per sampled pair. Median inference times are 17.56 ms
  left and 16.80 ms right; no configured reduced processing rate for the right.
- Raw detection logging happens independently of downstream ownership matching.
- Capture and inference are sequential, left first. Uncalibrated pointer position
  uses the primary view, but that happens after detection. Neither fact is evidence
  of an established right-detector bug. The archive cannot replay camera drivers
  or raw inference, because frames were not saved.

All 4,520 recorded raw hand estimates were reclassified from their saved world
landmarks with zero discrepancies (2,632 left / 1,888 right). For all 1,581
samples containing both selected hands, reversing the hand order in the shared
fusion function changed neither availability nor gesture. These checks establish
classifier consistency and order invariance on supplied landmarks, not visual
model accuracy or capture-order independence.

No unequal per-view detection configuration was found. Camera/position/lighting,
focus, occlusion, model view sensitivity, driver buffering and capture order
remain possible contributors. There are no images or exposure/focus/brightness
measurements to distinguish them. A logical swap of --left and --right, keeping
both physical cameras and lighting in place, is now a justified next comparison:
does the weak view follow physical index 2 or the secondary slot? Repeat the same
poses at the same distance. A weakness following a device/view still does not
separate hardware from that view's lighting, placement or occlusion.

## Calibration and approximately ten feet

The user reports sitting about ten feet away (about 3.05 m); the uncalibrated log
does not independently measure that distance. Calibration estimates lens and
stereo geometry for metric reconstruction and geometric hand-pair validation.
The app still runs the same hand detector on raw images first, then uses the
calibration for pairing/reconstruction. It does not calibrate or retrain the
detector, enlarge a hand in the image, change focus/exposure, or recover a hand
that the detector did not return. Therefore calibration is useful for stereo
positioning and geometric matching, but is not a remedy for these missing raw
detections or proof of reliable operation at ten feet.

References: [MediaPipe Hand Landmarker configuration and video behavior](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python),
[OpenCV camera calibration and reconstruction](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html).

## Validation and limits

ZIP CRC check passed. Counts were independently aggregated from samples.jsonl and
cross-checked against scripts/analyze_debug.py and summary.json. Reclassified all
raw landmarks and replayed fusion in both orders as described above; all assertions
passed. No desktop input, new model inference, camera acquisition or changes to
recognition thresholds occurred. Full application tests were not rerun for this
analysis/documentation-only change. B14 and B10 remain open.

## Camera-order swap follow-up

User supplied `debug-20261022T022911Z-4471e402.zip` after swapping the logical
camera order without reporting a physical camera move. SHA-256:
`84d04cb9220a28e6974e82f8bf8cc31f24018b008c1f23950161023129337d6a`.
The archive records camera indices [2, 1], revision 8faceeb, both MSMF streams at
1920x1080 / 30 reported FPS, and uncalibrated testing. Its 1,516 samples cover
elapsed 7.266–122.937 seconds, about 13.10 processed pairs/sec.

| Measure | Left: physical index 2 | Right: physical index 1 |
| --- | ---: | ---: |
| Samples with any raw hand | 1,002 / 1,516 (66.09%) | 1,186 / 1,516 (78.23%) |
| Named gesture estimates / raw hand detections | 874 / 1,349 (64.79%) | 1,202 / 1,832 (65.61%) |
| Samples with a named gesture | 859 / 1,516 (56.66%) | 1,016 / 1,516 (67.02%) |

Both views detect hands in 883 samples; left-only 119, right-only 303 and neither
211. Thus index 2 remains the weaker detector after becoming the primary/first
view. This supports the camera/view hypothesis over an invariant right/second-slot
problem. The sessions are not identical labelled stimuli, so this is evidence,
not proof that capture order has no influence. Lighting, lens focus, viewpoint,
occlusion and camera hardware are not distinguished by these archives.

The swapped session also has many multi-hand detections. During its 941 owned
samples, the selected owner is absent despite other detections in 127 left and
116 right samples. Lowering detection thresholds only when a view has zero raw
hands cannot address all owner-association failures. Pair timing rejects remain
zero; this still does not prove synchronized camera exposures.

ZIP CRC check passed. Independently counted raw availability and named gestures
from samples.jsonl and cross-checked the existing analyzer. No image inference
can be replayed from this data-only archive. Feature 13 records the user-approved
debug-only adaptive reacquisition experiment; its live benefit is unvalidated.
