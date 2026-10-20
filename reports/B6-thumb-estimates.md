# B6: thumb estimate disagrees with reported posture

The user reports keeping the thumb tucked during scrolling experiments. Earlier
wording such as “relax thumb” described an allowed pose, not an observed action.
An open/right-click classifier label is a **model-derived estimate**, not proof
that the user opened or relaxed their thumb. The wrist instruction now says that
thumb estimates are ignored.

Recomputed with classifier version 2 from the four supplied archives:

| Session (UTC suffix) | All detected: tucked / open / uncertain | Recorded live: tucked / open / uncertain |
| --- | --- | --- |
| Finger 200538Z | 4199 / 122 / 53 | 1106 / 0 / 0 |
| Finger 200823Z | 2863 / 174 / 188 | 942 / 7 / 13 |
| Wrist 202052Z | 613 / 577 / 103 | 233 / 24 / 18 |
| Wrist 203236Z | 839 / 241 / 23 | 647 / 159 / 9 |

“Live” here is the recorded controller phase, including invalid-shape frames;
these counts are not intended gesture accuracy. The old strict-thumb experiment
could reject the very frames needed to calibrate, so recorded live counts alone
understate the original blocker. See B5 for that analysis.

The classifier estimates open when model thumb IP angle exceeds 145 degrees and
thumb-tip/index-MCP separation exceeds 0.30 palm lengths. It estimates tucked
below 140 degrees or below that separation threshold; boundary cases are uncertain.
The supplied world landmarks cross these thresholds. Archives contain neither
images nor per-frame intended-thumb labels, so they cannot isolate incorrect
landmarks under foreshortening/occlusion from a poorly separating geometric rule.
Changing that global rule using these unlabelled sessions could affect ordinary
left/right clicks. This investigation does not mark B6 fixed.

Feature 7 now ignores thumb state during latched scrolling and excludes thumb
points from stereo reconstruction in that mode. The UI calls gesture labels
“pose estimates” and explicitly says clicks are ignored while scrolling. A thumb
estimate still participates in initial entry versus right-click selection, before
scroll mode is latched. Ordinary click classification remains unchanged.

Reproduce aggregate counts and the isolated main-engine replay:

```sh
python scripts/replay_main_scroll.py diagnostics/tilt-20261020T200538Z-01a0d29f.zip diagnostics/tilt-20261020T200823Z-18b8701c.zip diagnostics/wrist-tilt-20261020T202052Z-dc822480.zip diagnostics/wrist-tilt-20261020T203236Z-0dbdfdbb.zip --replay diagnostics/wrist-tilt-20261020T203236Z-0dbdfdbb.zip
```

See `7-scroll-replay.json` for checksums and exact aggregates. No raw hand data or
images are committed, and the replay emits no OS input.
