# Bug ledger

Record reported bugs and bugs incidentally noticed during authorized work.
Recording a bug does not authorize fixing it. Keep resolved entries for history.

| Bug number | Description | Related feature | Fixed? | Resolution / validation |
| --- | --- | --- | --- | --- |
| B1 | Camera startup exits with generic guidance while macOS first-use authorization is pending or denied. | 2, 3 | Yes (diagnostics); OS permission requires user action | B1: Reported permission failure; added macOS-specific Allow/rerun, Camera settings and launcher restart instructions. Regression suite: 33 passed, 1 Windows-only check skipped; camera cleanup preserved. Granting permission requires user action and has not been verified on the user’s launcher. |
| B2 | Real-hand pointing is rotation-sensitive; left click intermittent; scroll, right click and full-palm stop fail to recognize. | 3, 5, 6, 7, 8 | No—fix implemented and replay-validated; live confirmation pending | Supplied 2,592-frame session exposed joint-angle/thumb-spacing rejection. Corrected classifier: 14.4% → 98.5% replay match; 30/30 clips reach dwell. 39 uncertain right-rolled pointing frames remain. 42 local tests passed, one Windows-only skip. Windows/macOS CI also passed on c59a8cf. See reports/B2-analysis.md. |
| B3 | Deep downward tilt becomes fist-like and pauses the experiment. | 10 | No | User report confirmed as finger-chain rejection in supplied sessions; bounded comfortable endpoints proposed, with pause retained for lost/folded poses. Design choice pending. |
| B4 | Tilt experiment applies the down range to upscroll, causing slow upscroll when backward range is limited. | 10 | No | First calibration: 49.10-degree down span vs 13.61-degree up movement, about 2.34 vs 12 lines/sec. Endpoint-midpoint or independent gain calibration proposed; user choice pending. |

Next bug number: **B5**.
