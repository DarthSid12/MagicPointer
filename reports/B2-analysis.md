# B2: recorded gesture recognition correction

The October 20, 2026 supplied session contains 2,592 detected-hand frames in 30
complete clips (six poses, five angles). Tracking succeeded throughout; most
rejection occurred in the deterministic gesture classifier.

The original classifier required an index DIP angle over 150 degrees and PIP over
155 degrees. Recorded extended index DIP angles had a median near 147 degrees;
extended middle fingers also sometimes fell below the PIP threshold. Thumb/index
spacing overlapped heavily: curled thumbs could be farther from the index than
extended thumbs, so the original 0.55/0.65 spacing gates rejected valid poses and
sometimes interpreted right-click poses as scrolling.

The correction uses fingertip reach beyond knuckles and the unfolded fraction of
the finger chain, allowing natural joint bends. Thumb IP bend is the primary
thumb-state cue, with a smaller spacing guard for a straight thumb lying across
the palm. The uncertain state and existing engine dwell/click latches remain.

| Expected pose | Recorded classifier | Corrected classifier | Frames |
| --- | --- | --- | --- |
| fist | 62.6% | 100.0% | 436 |
| point | 22.9% | 91.1% | 437 |
| left click | 0.0% | 100.0% | 455 |
| scroll | 0.0% | 100.0% | 454 |
| right click | 0.0% | 100.0% | 452 |
| open palm | 0.0% | 100.0% | 358 |

Overall: **373/2,592 (14.4%) → 2,553/2,592 (98.5%)**. All 30 clips contain
correct continuous runs meeting the engine dwell requirement (0.45 seconds for
fist, 0.18 seconds for other gestures). No frame in this labelled session changes
to a different recognized action pose; the remaining errors are unrecognized.
This is not a false-click rate measurement: the session does not include natural
transitions, random poses or other users.

The 39 remaining uncertain frames occur in the right-rolled pointing clip, where
the model estimates a strongly folded index chain. That clip matches 51/90 frames
and its longest correct run is about 0.434 seconds. These frames are deliberately
not forced into a recognized pose. A live retest is needed to assess remaining
tracking uncertainty and practical activation/click/scroll continuity.

This session informed the correction; the replay score is **not independent test
accuracy**. Other hands, room distances, occlusion, transitions and stereo use are
still unverified. Only aggregate results are committed; the original landmark
archive remains in the ignored diagnostics directory.

The JSON report includes the source archive checksum, per-clip counts and longest
correct runs. Reproduce it with:

```sh
python scripts/evaluate_gestures.py diagnostics/gesture-20261020T193733Z-63f477c6.zip --output reports/B2-replay.json
```

Local regression validation: 43 tests, 42 passed, one Windows-only ABI check
skipped. New synthetic regressions cover naturally bent fingers, narrow open
thumbs, wide bent thumbs, ambiguity rejection and replay scoring. Tests continue
to rotate and scale poses, and the existing engine timing/action tests pass.
