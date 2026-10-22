# B9 / B8: unreliable owner control in the supplied debug session

Source: debug-20261021T234500Z-bd89e971.zip. The committed JSON report contains its
SHA-256 and aggregates, without raw landmarks. The user reports problems with
both selection and gesture actions. The session has no per-frame intended-gesture
labels or images, so estimated poses cannot establish actual gesture accuracy.

## What the recording establishes

There are 1,086 frames over 142.34 seconds, about 7.75 processing FPS despite the
cameras reporting 30 FPS. Median loop interval is 130.15 ms. The pair timestamp
check rejects 105 frames. The first activation occurs at 69.85 seconds. All five
activations end in tracking-loss deactivation; four failures follow one rejected
pair and a roughly 0.32-0.36-second gap between accepted observations. At this
processing rate, a single rejected frame can exceed the 0.30-second timeout.
That forced another fist activation while the owner ID remained locked.

The recorded owner pair [15, 14] persists for 413 frames. In 177 frames it is not
matched in both views. Cameras 0/1 have 132/134 owned frames where other detections
exist but the selected ID is unavailable. These are not all simple detection
misses. They can also include model duplicates, ambiguous matches, stale position,
hand changes, or handedness instability. The old archive does not contain the
assignment cost matrix needed to distinguish them definitively.

At 103.17 seconds, the tracked palm is close to its previous position, but the
secondary view's handedness estimate changes from Right to Left (score 0.865).
The old hard label gate rejects continuity and creates another ID. Both raw
estimates identify an open palm in that frame. This does not prove physical
identity, but exposes a concrete label-only tracking rejection at a nearby pose.

Six frames where both sole camera estimates say left click occur at 108.87,
111.31, 112.08, 112.87, 113.00 and 114.29 seconds. All are blocked by the stale
owner match. Whether those are the originally selected physical hand cannot be
established from these recordings. There are no emitted clicks or wheel events
in the session. Relaxing gesture classification alone would not bypass that gate.

## Corrections

- Main control retains activation through missing/rejected views, pauses input
  immediately, and resumes the same selected IDs without another fist/beep.
  There is still no one-camera fallback or relaxed pair timing. Recovery clears
  pending click/scroll state, rebases pointing without a jump and requires fresh
  neutral for scrolling. Point/fist rearming prevents a held click chord from
  firing on recovery. A processing stall receives the same pause treatment.
- Initial fist selection counts 2.5 seconds of valid evidence. Brief unknown or
  missing samples up to 0.4 seconds pause rather than erase evidence; invalid
  intervals are not credited. A known different gesture, a different hand ID or
  a longer interruption resets selection. This targets repeated countdown resets.
- An already bound owner may keep its ID through a label flip only within 0.75
  palm lengths, with a soft label penalty and the existing uniqueness checks.
  Unbound pairs still require agreeing handedness; remote or ambiguous hands do
  not replace the owner. Fusion can tolerate label disagreement only for already
  bound IDs. No broad position-gate widening or global hand reassignment was added.
- User confirmed immediate release for one confidently matched frame where both
  per-view palm estimates and fused palm agree. Otherwise the 0.18-second hold
  remains. An invalid camera pair cannot release the owner from stale evidence.

## Isolated replay

Both controllers are seeded at the same first recorded activation, then receive
the exact same selected observations and timestamps. The legacy policy reproduces
all recorded events exactly: 5 activations, 58 moves and 5 tracking-loss stops.
With owner retention, there are zero tracking-loss stops; observed palm gestures
produce 3 palm stops, with 4 activations and 117 moves. Activation grace is disabled
in both replay policies so that only pause/recovery behavior is compared.

This is a control-logic comparison, not a full rerun of detection, tracking,
selection, new palm consensus or gesture intent. Increased output is not accuracy.
Synthetic tests separately cover label continuity, missing-view suppression,
recovery clicks/scroll, activation gaps, identity changes and two-view palm release.

Reproduce:

```sh
python scripts/analyze_debug.py diagnostics/debug-20261021T234500Z-bd89e971.zip --output reports/B9-debug-analysis.json
```

## Expanded future diagnostics (11.1)

New debug archives use schema 2 and retain data-only recording. Each sample adds
local track history and all candidate costs/rejections/assignments, all camera
pair matching results, per-landmark and per-finger shape disagreement, palm release
evidence, pre/post engine state, activation evidence and pause/rebase/click-latch
state. Per-stage timing separates capture, each camera's inference, selection,
observation checks and engine work; logging/render times refer to the previous
frame to avoid logging unmeasured work. Summary includes status counts and median,
p95 and maximum timing. Hand pixel extents and palm size, model checksum, package
versions, source revision and tracked-file dirty flag support reproduction.

No images, video, audio, environment variables or credentials are recorded. Git
revision may be unavailable outside a checkout. The extra telemetry does not
provide ground truth or invent model landmark confidence. Live recognition and
owner continuity still need a new user session; B8 remains open for confirmation.
