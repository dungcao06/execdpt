# ExecDPT

**In-Context Reinforcement Learning for Optimal Trade Execution**

> Working paper targeting ICAIF 2026 (ACM International Conference on AI in Finance)

## Overview

ExecDPT applies Decision-Pretrained Transformers (Lee et al., NeurIPS 2023) to optimal trade execution. A single transformer is pretrained across thousands of (stock, date, parent-size) execution MDPs and deployed zero-shot on unseen assets, regimes, and order sizes — with no gradient updates, only in-context inference.

## Research Questions

1. **RQ1 (Core):** Can ExecDPT match per-task PPO on unseen stocks with zero gradient updates?
2. **RQ2 (Generalization):** Does in-context adaptation degrade more gracefully under regime shifts?
3. **RQ3 (Mechanism):** Is the model genuinely inferring task properties in context?

## Repository Structure

```
execdpt/
├── simulator/       # JaxMARL-HFT / ABIDES environment wrappers
├── experts/         # Per-task PPO training + Almgren-Chriss baseline
├── dpt/             # ExecDPT architecture, pretraining, inference
├── eval/            # Evaluation harness, metrics, statistical tests
├── figures/         # Reproduction scripts for all paper figures
├── configs/         # Task distribution, splits, hyperparameters
├── paper/           # LaTeX source (ACM sigconf)
└── scripts/         # Utility scripts (data download, preprocessing)
```

## Pre-registered Train/Test Splits

> **⚠️ FROZEN BEFORE ANY MODEL TRAINING — DO NOT MODIFY**
>
> Timestamp of registration: see git log for this commit.

### Task Distribution

| Dimension | Values |
|-----------|--------|
| Tickers | 100 NASDAQ from LOBSTER, stratified by market cap & sector |
| Dates | 50 randomly sampled trading days per ticker, Jan 2018 – Mar 2024 |
| Parent sizes | {0.5%, 1%, 2%, 5%, 10%} of ADV |
| Horizons | {30 min, 1 hr, full day} |
| Sides | Buy and sell, balanced |

### Held-out Evaluation Splits

| Split | Definition | Tests |
|-------|-----------|-------|
| **Stock-OOD** | 20 tickers never seen during pretraining | Cross-asset transfer |
| **Date-OOD** | All tasks dated Apr 2024 – Mar 2026 | Regime-shift robustness |
| **Size-OOD** | Parent sizes ∈ {15%, 25%} ADV | Out-of-distribution sizes |
| **Joint-OOD** | All three dimensions held out | Hardest test |

### Ticker Split

See `configs/splits.yaml` for the exact ticker lists.

- **Pretraining tickers:** 80 NASDAQ tickers (large/mid/small cap, diversified sectors)
- **Stock-OOD tickers:** 20 held-out NASDAQ tickers (same stratification)

### Date Boundary

- **Training window:** January 2, 2018 – March 29, 2024
- **Date-OOD window:** April 1, 2024 – March 31, 2026

## Setup

```bash
# Create conda environment
conda create -n execdpt python=3.11 -y
conda activate execdpt

# Install dependencies
pip install -r requirements.txt
```

## License

Private until acceptance. Will be open-sourced under MIT on submission day.
