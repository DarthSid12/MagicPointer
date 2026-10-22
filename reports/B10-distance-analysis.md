# B10: preliminary distance-related detection investigation

Source: diagnostics/debug-20261022T001115Z-1db11901.zip, explicitly identified
by the user. Data only; no images or measured hand-to-camera distances.

The 671 samples span 69.07 seconds. Both cameras negotiated 1920 x 1080 at
approximately 30 reported FPS; processing is about 9.7 FPS.

| Raw detector availability | Frames |
| --- | ---: |
| Neither camera | 339 |
| Primary only | 75 |
| Secondary only | 254 |
| Both cameras | 3 |

Thus primary detection is present in 78 frames and secondary in 257. These
counts include every raw hand, before local identity or cross-view matching.
The shared controller cannot operate in most frames under the user-required
two-view agreement policy. There are also 31 pair timing rejects, but those
do not account for the absent raw detections. No activation occurs.

For detected hands, wrist-to-middle-knuckle image distance has a median of
77.1 pixels in the primary view and 33.4 pixels in the secondary view.
Ranges are 28.7-108.8 and 12.1-101.2 pixels respectively. The secondary produces
248 unrecognized estimates out of 270 hand detections. Primary estimates include
73 palms out of 100 hand detections. Hand-detection totals differ from frame
counts because more than one hand can appear. Actual intended gestures are unknown.

Across elapsed 20-30 seconds neither view detects a hand in any of 99 samples.
Across 40-60 seconds secondary-only detections dominate (182 of 192 samples).
This supports investigating small-hand detection and view-dependent failures
before changing gesture thresholds or matching. It does not prove that distance
alone caused every missing detection: visibility, focus, lighting, pose and
model misses cannot be distinguished without additional evidence. Pixel size
is measured only while a hand is detected; there is no size measurement during
a missing detection. World hand landmarks are not calibrated room distances.

The latest tested application was merged and pushed to main first, at d50562f.
No detector settings or input policies were changed in this checkpoint. B10
remains open; this analysis does not establish reliable 12-foot operation.
