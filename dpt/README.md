# DPT
ExecDPT architecture, pretraining pipeline, and inference.

## Contents
- `model.py` — Decoder-only transformer (DPT architecture)
- `tokenizer.py` — Transition (s, a, r, s') → token encoder
- `dataset.py` — Trajectory dataset loader
- `train.py` — Supervised pretraining loop (cross-entropy on expert actions)
- `inference.py` — Zero-shot and in-context evaluation interface
