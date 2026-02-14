# Contributing

Thanks for contributing.

## Prerequisites

- Python 3.11+
- `venv`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Standards

- Keep dataset rows in JSONL format:
  - `{"text": "...", "label": 0|1, "category": "..."}`
- Use `category="normal"` for non-spam rows.
- Keep spam categories consistent (`phishing`, `job_scam`, `crypto`, `adult`, `giveaway`, `marketing`, `spam`).
- Avoid adding low-quality synthetic rows or noisy duplicates.

## Required Checks

Run these before opening a PR:

```bash
source .venv/bin/activate
python tests/data_quality_lint.py
python train.py
python tests/build_eval_suite.py
python tests/evaluate.py
```

PRs should pass `QUALITY GATE: PASS`.

## Suggested PR Structure

- Small, focused changes
- Clear commit messages
- Include benchmark impact when model/data changes
- If data was added, explain source format and filtering quality rules

## Reporting Issues

Please include:

- Reproduction input text
- Expected vs actual output
- Current model metadata (`python model.py`)
- Relevant command output from lint/eval
