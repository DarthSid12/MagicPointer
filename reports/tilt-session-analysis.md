# Tilt experiment: user session analysis

The two supplied sessions were analyzed as data, without executing archive
contents or changing the scroll controller. Raw recordings remain in diagnostics/.
Aggregate counts and source checksums are in tilt-session-analysis.json.

| Session | Frames | Valid experimental shape | No hand | Finger-chain rejection | Thumb/ring/little rejection | Axis disagreement |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 4639 | 4039 | 265 | 157 | 178 | 0 |
| 2 | 3903 | 2774 | 678 | 83 | 364 | 4 |

## B3: deep tilt becomes fist-like

There were 154 and 74 invalid frames labelled fist in the two sessions. The
experimental shape check rejects finger chains that are no longer sufficiently
extended. The first session's rejected fist-like frames often sit near that
boundary; the second also includes strongly folded estimates. The ordinary
classifier's fist label does not itself stop the experiment; its separate
geometry check does. The logs cannot establish whether every rejection was an
intentional fist, excessive flexion or a landmark estimation error.

Continuing scrolling through a fist-shaped observation would make intentional
release ambiguous. A bounded comfortable calibration should reach full speed
before the fingers approach that range. Greater valid tilt should remain capped
at full speed; genuinely lost/folded poses should pause rather than continue a
stale command. A live test of shallower endpoints is required to validate this
mitigation; no threshold loosening is justified by the unlabeled failure frames.

## B4: symmetric gain ignores asymmetric mobility

The current controller uses the captured DOWN span for BOTH directions. For the
first session's first live calibration, down span was 49.10 degrees and maximum
recorded up offset was only 13.61 degrees. With the 5-degree dead zone, upscroll
therefore reached about 2.34 lines/sec against the 12-line/sec cap. Later changed
calibrations reached faster upscroll; combining all frames would conceal this
per-calibration problem. The second session also demonstrates different gains
across calibration attempts.

Recommended experiment revision: capture comfortable straight/less-flexed UP and
modestly flexed DOWN endpoints, place neutral halfway between, and map each end
to full speed. The neutral pose is then partially flexed, so UP requires
straightening instead of backward hyperextension. An alternative is retaining a
chosen neutral and independently calibrating both directions. That alternative
still requires some usable travel on each side of neutral. User preference is
pending before implementing either behavior.

## Limits

These sessions are unlabelled free exploration. Recorded speeds measure the
controller output, not intended direction accuracy or comfort. Missing hands and
pose changes may include intentional releases or reaching for calibration keys.
No changes to production scrolling, recognition thresholds or experimental
calibration are part of this analysis. No code tests are needed for this report;
aggregate figures were derived from the JSONL samples and whitespace was checked.
