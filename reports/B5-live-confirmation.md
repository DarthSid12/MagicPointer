# B5: successful user retest

The user supplied wrist-tilt-20261020T203236Z-0dbdfdbb.zip and reported that the
experiment looks good. It contains 1,522 frames over approximately 53.4 seconds,
with the corrected relaxed-thumb wrist policy, a 10-degree neutral dead zone and
30-degree full-speed angle. Raw landmarks remain in ignored diagnostics/.

## Observed controller behavior

- Both neutral captures succeeded (about 17.8 and 31.7 seconds into the session).
- Live mode contains 815 frames: 799 valid (98.04%) and 16 invalid. All 16 live
  rejections are extended-finger-chain checks; none are thumb-related.
- Valid live outputs: 275 down-command frames, 288 up-command frames and 236 stop
  frames. Both directions reach the configured 12 simulated lines/second cap.
- Longest uninterrupted command runs: down 7.43 seconds, up 6.84 seconds and
  neutral stop 2.27 seconds. These runs exclude invalid/non-live frames and gaps
  greater than 0.30 seconds.
- Replaying the recorded landmarks, timestamps and key events with the current
  controller exactly reproduces valid/live counts, direction counts, maximum
  speeds, rejection totals and captured neutral angles.

The full session also contains setup/release periods: 419 no-hand frames, 116
finger-chain rejections and 7 ring/little rejections. Most are outside live mode;
without frame-by-frame intent labels, they should not be counted as failed scroll
attempts. Likewise, the 98.04% figure is valid tracking/shape during live mode,
not intended-gesture accuracy or an independent population estimate.

## Outcome and limits

The user report and observed sustained up/down output support confirming the B5
thumb-gate correction for this live trial. No controller or threshold changes are
needed from this recording. No runtime code was changed in this confirmation.
Only the existing replay analysis and whitespace checks were run; the unchanged
software retains its previously recorded automated/CI results.

This is one user's one-camera macOS experiment. It does not establish 12-foot
reliability, other users/camera setups, native Windows camera behavior or real OS
scroll performance. The regular pointer controller still uses its existing
movement-based scroll behavior; wrist scrolling remains an isolated experiment.

See B5-confirmation-replay.json for aggregate replay statistics and the source
checksum. Reproduce with:

```sh
python scripts/replay_wrist_tilt.py diagnostics/wrist-tilt-20261020T203236Z-0dbdfdbb.zip --output reports/B5-confirmation-replay.json
```
