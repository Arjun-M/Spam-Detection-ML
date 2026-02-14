# Spam Detection ML

Repository: `Arjun-M/Spam-Detection-ML`

Production-oriented SMS and chat spam detection built with Python, scikit-learn, TF-IDF, and logistic regression.

![CI](https://github.com/Arjun-M/Spam-Detection-ML/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/github/license/Arjun-M/Spam-Detection-ML)
![Repo Size](https://img.shields.io/github/repo-size/Arjun-M/Spam-Detection-ML)

## Features

- Binary spam detection (`spam` vs `normal`)
- Spam category classification (`phishing`, `job_scam`, `crypto`, `adult`, `giveaway`, `marketing`, `spam`)
- Long-message chunking for paragraph-level inference
- Data quality linting and deterministic benchmark quality gate
- Lightweight model artifacts for practical deployment

## Why This Repo

This project provides a practical spam stack that is easy to run locally and deploy in lightweight services:
- Strong spam precision for moderation pipelines
- Category-level routing for security and abuse workflows
- Guardrails for short-message false-positive control
- Reproducible local training and evaluation workflow

## Architecture

1. Text preprocessing in `utils.py`
2. Hybrid TF-IDF feature extraction (word + character n-grams)
3. Binary logistic regression with tuned threshold
4. Spam-category logistic regression for positive messages
5. Inference guardrails in `model.py` for known edge cases

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
python tests/build_eval_suite.py
python tests/evaluate.py
```

## Inference Example

```python
from model import run_model

result = run_model("verify your account now")
print(result)
```

Output fields:
- `is_spam`
- `confidence`
- `category`
- `threshold_used`
- `chunked` / `chunk_count` (for long messages)

## Current Performance Snapshot

Latest local benchmark run (`tests/evaluate.py`):
- Accuracy: `0.9769`
- Precision: `1.0000`
- Recall: `0.9626`
- F1: `0.9810`
- Spam-category accuracy: `0.9486`
- Quality gate: `PASS`

## Model Footprint

Current artifact size in `models/`:
- Total: about `2.2 MB`
- `binary_model.pkl`: `181,355 bytes`
- `category_model.pkl`: `1,264,399 bytes`
- `vectorizer.pkl`: `821,907 bytes`
- `metadata.json`: `195 bytes`

## Repository Structure

- `train.py`: training pipeline
- `model.py`: inference runtime
- `main.py`: interactive CLI
- `config.py`: model and runtime settings
- `utils.py`: preprocessing helpers
- `dataset/`: JSONL datasets
- `models/`: trained artifacts
- `tests/data_quality_lint.py`: dataset validation
- `tests/build_eval_suite.py`: deterministic eval suite generation
- `tests/evaluate.py`: benchmark and quality gate
- `tests/convert_text_labels_to_jsonl.py`: `label,text` to JSONL converter

## Dataset Schema

All dataset rows use JSONL:

```json
{"text": "message text", "label": 0, "category": "normal"}
```

Rules:
- `label=0` means non-spam
- `label=1` means spam
- Non-spam rows should use `category="normal"`
- Spam rows should use specific categories (for example `phishing`, `crypto`, `spam`)

## Data Conversion (`label,text` to JSONL)

```bash
source .venv/bin/activate
python tests/convert_text_labels_to_jsonl.py \
  --input spam.txt \
  --out-dir dataset \
  --ham-category normal \
  --spam-category spam
```

## Open Source Workflow

1. Add or update dataset rows in `dataset/*.jsonl`
2. Run lint: `python tests/data_quality_lint.py`
3. Train: `python train.py`
4. Build benchmark suite: `python tests/build_eval_suite.py`
5. Validate: `python tests/evaluate.py`
6. Open a PR with metrics impact notes

## Link Domain Policy

- Unknown domains are treated as normal by default.
- Domains listed in `config.py` as `BLOCKED_URL_DOMAINS` are treated as spam.
- Messages with explicit scam cues (`verify account`, `claim prize`, and similar) can still be flagged.

## Data Note

This project supports the widely used public SMS spam corpus format commonly mirrored in Kaggle datasets, along with additional chat-style and edge-case examples.

## Contributing

Contribution workflow is documented in `CONTRIBUTING.md`.
