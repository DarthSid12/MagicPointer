# B5: wrist downscroll thumb blocker

The supplied session contains 1,952 frames: 1,293 with a detected hand and 659
without one. The previous wrist experiment accepted 586 frames. Its first shape
check rejected 681 frames for thumb/ring/little state: the thumb was a blocker in
680, and was the only finger-state blocker in 642. Many downward-looking poses
had an open or uncertain model thumb estimate. This check prevented the camera
pitch and speed mapping from running at all.

## Correction

Only the explicitly calibrated, no-click wrist experiment now ignores thumb
position. Index/middle must still form extended, aligned chains; ring/little must
remain curled; palm geometry and camera direction must be usable. Lost/invalid
tracking still pauses immediately and disarms after the existing timeout. The
normal mouse gesture classifier and original finger-tilt experiment retain their
thumb rules. The wrist UI and logs document the relaxed-thumb policy. A normal
classifier label of `right click` can coexist with valid wrist scrolling in this
preview; it never sends clicks or OS input.

## Replay

The current controller was fed recorded landmarks and timestamps, then the
recorded neutral/reset key events in their original order. Failed capture
attempts are retried under the corrected validity history, just as a user pressing
the same keys would experience. Baseline statistics come from recorded outputs.
No cameras or desktop events are used. Only aggregates are committed.

| Measurement | Recorded | Replayed after correction |
| --- | --- | --- |
| Accepted shape/direction frames | 586 | 938 |
| Live-phase frames | 275 | 675 |
| Down command frames | 2 | 96 |
| Up command frames | 105 | 172 |
| Maximum down speed (lines/sec) | 2.09 | 12.00 |

Residual replay rejections: 659 no-hand frames, 213 extended-finger-chain
failures, 100 camera-direction failures, 39 ring/little failures and 3 finger-axis
disagreements. Thumb-only exclusion was an early gate, so removing it exposes
these other checks rather than making every rejected pose acceptable.

This is an unlabelled free-exploration recording. Command counts measure controller
output, **not intended-direction accuracy**. It does not distinguish all intentional
releases from model/occlusion failures. The correction restores downscroll output
on this replay but does not establish that all real downward poses track reliably.
Live confirmation remains necessary; residual rejection should not be bypassed
without labelled evidence.

Reproduce with:

```sh
python scripts/replay_wrist_tilt.py diagnostics/wrist-tilt-20261020T202052Z-dc822480.zip --output reports/B5-wrist-replay.json
```

The JSON report includes a source checksum and neutral-capture outcomes. Local
regression validation: 62 tests, 61 passed, one Windows-only ABI check skipped.
New tests exercise open-thumb downscroll, preserved global/finger-mode thumb rules,
continued fist/palm rejection, and recorded key-event replay without raw-data
publication. A rendered open-thumb wrist preview is also visually checked.
