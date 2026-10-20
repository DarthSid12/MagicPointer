# Feature ledger

Numbers identify major prompts. Updates use `X.Y`; revamps reuse the original
`X`. Preserve history and use the latest update count to allocate the next `Y`.

| Feature number | Feature / description | Status | Latest update count | Prompt history |
| --- | --- | --- | --- | --- |

| 1 | Windows-first application foundation and startup | Planned; development decisions pending | 0 | 1: Initial request; feature breakdown and validation plan recorded. |
| 2 | Two-camera acquisition and startup stereo calibration | Planned; camera hardware and calibration method pending | 0 | 2: Initial request; fixed stereo rig and calibration required. |
| 3 | Hand tracking and single-camera visual debug mode | Planned | 0 | 3: Initial request; hand box and state colors; works with one camera. |
| 4 | Raised closed-fist activation and activation chime | Planned; activation meaning pending | 0 | 4: Initial request; capture activation origin and center cursor. |
| 5 | Index-tip pointer movement with depth compensation | Planned; pointer mapping pending | 0 | 5: Initial request; track index tip relative to activation origin. |
| 6 | Thumb left click and thumb-plus-middle right click | Planned | 0 | 6: Initial request; mutually exclusive, debounced click gestures. |
| 7 | Index-plus-middle scrolling | Planned | 0 | 7: Initial request; two-finger motion scrolls. |
| 8 | Full-palm deactivation and deactivation chime | Planned | 0 | 8: Initial request; stop control and await another activation. |

Next major prompt number: **9**.

This request deliberately allocates separate numbers to its features as directed
by the user. No runtime functionality has been implemented or validated yet.
See DEVELOPMENT.md for dependencies, acceptance checks, and open decisions.
