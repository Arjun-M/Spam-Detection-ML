import json
import random
from pathlib import Path

DATASET_DIR = Path("dataset")
OUTPUT_PATH = DATASET_DIR / "eval_suite.jsonl"
SEED = 42

# Fixed per-category counts from existing labeled data.
COUNTS = {
    "normal": 120,
    "phishing": 50,
    "job_scam": 40,
    "giveaway": 40,
    "marketing": 30,
    "crypto": 20,
    "adult": 20,
}

# Real-world messaging edge cases and regressions.
MANUAL_CASES = [
    {"text": "lol", "label": 0, "category": "normal"},
    {"text": "ok", "label": 0, "category": "normal"},
    {"text": "new video", "label": 0, "category": "normal"},
    {"text": "join now", "label": 0, "category": "normal"},
    {"text": "i won football match today hurray", "label": 0, "category": "normal"},
    {"text": "we won the league finals today", "label": 0, "category": "normal"},
    {"text": "can you call me in 5", "label": 0, "category": "normal"},
    {"text": "i am on my way", "label": 0, "category": "normal"},
    {"text": "adult should be older than 18", "label": 0, "category": "normal"},
    {"text": "adult supervision is required in this lab", "label": 0, "category": "normal"},
    {"text": "please review pull request", "label": 0, "category": "normal"},
    {
        "text": "As per your request callertune has been set successfully for callers",
        "label": 0,
        "category": "normal",
    },
    {"text": "i have just won a lambo", "label": 1, "category": "giveaway"},
    {"text": "i have just won a lambo hurray join me now", "label": 1, "category": "giveaway"},
    {"text": "you won prize click to claim now", "label": 1, "category": "giveaway"},
    {"text": "claim your reward now", "label": 1, "category": "giveaway"},
    {"text": "verify your account now", "label": 1, "category": "phishing"},
    {"text": "urgent security alert verify identity", "label": 1, "category": "phishing"},
    {"text": "CEO urgent buy gift cards and share code", "label": 1, "category": "phishing"},
    {"text": "account suspended login now to restore access", "label": 1, "category": "phishing"},
    {"text": "work from home earn 5000 weekly", "label": 1, "category": "job_scam"},
    {"text": "no experience needed earn daily", "label": 1, "category": "job_scam"},
    {"text": "free crypto airdrop join telegram", "label": 1, "category": "crypto"},
    {"text": "buy btc now and get bonus", "label": 1, "category": "crypto"},
    {"text": "private adult chat click now", "label": 1, "category": "adult"},
    {"text": "new adult videos uploaded hourly", "label": 1, "category": "adult"},
]


def load_category_records(category, count, rng):
    records = []
    for path in sorted(DATASET_DIR.glob("*.jsonl")):
        if path.name == "eval_suite.jsonl":
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if str(row["category"]) != category:
                    continue
                records.append(
                    {
                        "text": row["text"],
                        "label": int(row["label"]),
                        "category": str(row["category"]),
                    }
                )

    if len(records) < count:
        raise ValueError(
            f"Not enough rows for category={category}: need {count}, got {len(records)}"
        )

    rng.shuffle(records)
    return records[:count]


def main():
    rng = random.Random(SEED)
    suite = []

    for category, count in COUNTS.items():
        suite.extend(load_category_records(category, count, rng))

    suite.extend(MANUAL_CASES)

    # Deduplicate by text while preserving first occurrence for deterministic output.
    deduped = []
    seen = set()
    for rec in suite:
        text = rec["text"].strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(rec)

    rng.shuffle(deduped)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for row in deduped:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    print(f"Wrote {len(deduped)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
