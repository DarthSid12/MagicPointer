# MagicPointer — feature 1 checkpoint

Windows-first pointer prototype. This checkpoint implements features 1–1.
Commands available: download-model. Later gesture behaviors are not implemented yet.
Hardware validation, native Windows input validation and the 12-foot target remain
pending. See FEATURES.md, DEVELOPMENT.md and VALIDATION.md.

Use 64-bit Python 3.11 on Windows. Create a virtual environment, then run:

```sh
python -m pip install -e .
python -m magicpointer --help
python -m magicpointer download-model
```
