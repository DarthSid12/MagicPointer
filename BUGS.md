# Bug ledger

Record reported bugs and bugs incidentally noticed during authorized work.
Recording a bug does not authorize fixing it. Keep resolved entries for history.

| Bug number | Description | Related feature | Fixed? | Resolution / validation |
| --- | --- | --- | --- | --- |

| B1 | Camera startup exits with generic guidance while macOS first-use authorization is pending or denied. | 2, 3 | Yes (diagnostics); OS permission requires user action | B1: Reported permission failure; added macOS-specific Allow/rerun, Camera settings and launcher restart instructions. Regression suite: 33 passed, 1 Windows-only check skipped; camera cleanup preserved. Granting permission requires user action and has not been verified on the user’s launcher. |

Next bug number: **B2**.
