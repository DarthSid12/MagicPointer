# Bug ledger

Record reported bugs and bugs incidentally noticed during authorized work.
Recording a bug does not authorize fixing it. Keep resolved entries for history.

| Bug number | Description | Related feature | Fixed? | Resolution / validation |
| --- | --- | --- | --- | --- |
| B1 | Camera startup exits with generic guidance while macOS first-use authorization is pending or denied. | 2, 3 | Yes (diagnostics); OS permission requires user action | B1: Reported permission failure; added macOS-specific Allow/rerun, Camera settings and launcher restart instructions. Regression suite: 33 passed, 1 Windows-only check skipped; camera cleanup preserved. Granting permission requires user action and has not been verified on the user’s launcher. |
| B2 | Real-hand pointing is rotation-sensitive; left click intermittent; scroll, right click and full-palm stop fail to recognize. | 3, 5, 6, 7, 8 | No—fix implemented and replay-validated; live confirmation pending | Supplied 2,592-frame session exposed joint-angle/thumb-spacing rejection. Corrected classifier: 14.4% → 98.5% replay match; 30/30 clips reach dwell. 39 uncertain right-rolled pointing frames remain. 42 local tests passed, one Windows-only skip. Windows/macOS CI also passed on c59a8cf. See reports/B2-analysis.md. |

Next bug number: **B3**.
