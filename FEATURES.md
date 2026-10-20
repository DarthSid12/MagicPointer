# Feature ledger

Each feature retains its original number; clarification of the initial request
does not increment update counters. Prototype status does not establish hardware
accuracy. See VALIDATION.md for completed checks and outstanding acceptance tests.

| Feature number | Feature / description | Status | Latest update count | Prompt history |
| --- | --- | --- | --- | --- |
| 1 | Windows-first application foundation and startup | Implemented prototype; Windows/macOS CI passes; physical rig validation pending | 0 | 1: Initial request and development plan; user confirmed camera/calibration and anywhere-in-view activation. Feature 1 implementation checkpoint validated and committed in this initial development sequence. |
| 2 | Two-camera capture, checkerboard calibration and wall-marker verification | Implemented prototype; Windows/macOS CI passes; physical rig validation pending | 0 | 2: Initial request and development plan; user confirmed camera/calibration and anywhere-in-view activation. Feature 2 implementation checkpoint validated and committed in this initial development sequence. B1: Added actionable macOS first-use camera authorization recovery; regression checks passed. |
| 3 | Hand landmarks and single-camera visual debug mode | Real-hand recognition failures reported (B2); recordings needed | 0 | 3: Initial request and development plan; user confirmed camera/calibration and anywhere-in-view activation. Feature 3 implementation checkpoint validated and committed in this initial development sequence. B2: User reports rotation-sensitive pointing and unreliable clicks, scroll and palm stop; diagnostic collection added as feature 9. |
| 4 | Closed-fist activation anywhere in view, origin capture and activation chime | Implemented prototype; Windows/macOS CI passes; physical rig validation pending | 0 | 4: Initial request and development plan; user confirmed camera/calibration and anywhere-in-view activation. Feature 4 implementation checkpoint validated and committed in this initial development sequence. |
| 5 | Index-tip pointer movement in a local metric workspace | Real-hand recognition failures reported (B2); recordings needed | 0 | 5: Initial request and development plan; user confirmed camera/calibration and anywhere-in-view activation. Feature 5 implementation checkpoint validated and committed in this initial development sequence. B2: User reports rotation-sensitive pointing and unreliable clicks, scroll and palm stop; diagnostic collection added as feature 9. |
| 6 | Thumb left click and thumb-plus-middle right click | Real-hand recognition failures reported (B2); recordings needed | 0 | 6: Initial request and development plan; user confirmed camera/calibration and anywhere-in-view activation. Feature 6 implementation checkpoint validated and committed in this initial development sequence. B2: User reports rotation-sensitive pointing and unreliable clicks, scroll and palm stop; diagnostic collection added as feature 9. |
| 7 | Index-plus-middle vertical scrolling | Real-hand recognition failures reported (B2); recordings needed | 0 | 7: Initial request and development plan; user confirmed camera/calibration and anywhere-in-view activation. Feature 7 implementation checkpoint validated and committed in this initial development sequence. B2: User reports rotation-sensitive pointing and unreliable clicks, scroll and palm stop; diagnostic collection added as feature 9. |
| 8 | Full-palm deactivation and distinct deactivation chime | Real-hand recognition failures reported (B2); recordings needed | 0 | 8: Initial request and development plan; user confirmed camera/calibration and anywhere-in-view activation. Feature 8 implementation checkpoint validated and committed in this initial development sequence. B2: User reports rotation-sensitive pointing and unreliable clicks, scroll and palm stop; diagnostic collection added as feature 9. |
| 9 | Guided single-camera gesture diagnostic collection and local labelled archive | Implemented; local automated validation passed; user recordings pending | 0 | 9: User authorized recording gestures at different angles to diagnose B2. Six poses, five angles, landmark-only samples and per-pose recognition summaries. |

Next major prompt number: **10**.

This request allocates separate numbers to eight features as directed by the user.

Windows and macOS CI passed on implementation commit `0758819`; see VALIDATION.md for the run and exact checks.
