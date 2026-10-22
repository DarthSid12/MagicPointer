# Bug ledger

Record reported bugs and bugs incidentally noticed during authorized work.
Recording a bug does not authorize fixing it. Keep resolved entries for history.

| Bug number | Description | Related feature | Fixed? | Resolution / validation |
| --- | --- | --- | --- | --- |

| B1 | Camera startup exits with generic guidance while macOS first-use authorization is pending or denied. | 2, 3 | Yes (diagnostics); OS permission requires user action | B1: Reported permission failure; added macOS-specific Allow/rerun, Camera settings and launcher restart instructions. Regression suite: 33 passed, 1 Windows-only check skipped; camera cleanup preserved. Granting permission requires user action and has not been verified on the user’s launcher. |

| B2 | Gestures glitch and are frequently missed on real hands: three-state finger thresholds reject relaxed poses, single noisy frames restart hold timers, and stereo requires both cameras to agree per frame. | 4, 5, 6, 7, 8 | Yes (synthetic validation); hardware tuning pending | B2: Extension-ratio classifier with hysteresis and smoothing, stereo ratio fusion, 0.10 s dropout grace, debounced handedness, wrist/tip-only triangulation, One-Euro pointer filter and pre-pose click position. Suite: 44 tests, 43 passed, 1 Windows-only skipped. See VALIDATION.md. |

Next bug number: **B3**.
