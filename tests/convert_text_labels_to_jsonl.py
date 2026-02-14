import argparse
import csv
import json
from pathlib import Path


def normalize_label(raw: str):
    v = (raw or "").strip().lower()
    if v == "ham":
        return 0
    if v == "spam":
        return 1
    return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a labeled text CSV (label,text) into two JSONL dataset files "
            "using the project schema."
        )
    )
    parser.add_argument("--input", default="spam.txt", help="Path to source text file")
    parser.add_argument("--out-dir", default="dataset", help="Output dataset directory")
    parser.add_argument(
        "--spam-category",
        default="spam",
        help="Category value for spam-labeled rows",
    )
    parser.add_argument(
        "--ham-category",
        default="normal",
        help="Category value for ham-labeled rows",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ham_out = out_dir / "spam_ham.jsonl"
    spam_out = out_dir / "spam_spam.jsonl"

    ham_count = 0
    spam_count = 0
    skipped = 0

    # De-duplicate within each output bucket by lowercased text.
    seen_ham = set()
    seen_spam = set()

    with in_path.open("r", encoding="utf-8", errors="replace", newline="") as src, \
        ham_out.open("w", encoding="utf-8") as ham_f, \
        spam_out.open("w", encoding="utf-8") as spam_f:

        reader = csv.reader(src)
        header_consumed = False

        for row in reader:
            if not row:
                continue

            # Skip header once.
            if not header_consumed:
                first = row[0].strip().lower() if row else ""
                second = row[1].strip().lower() if len(row) > 1 else ""
                if first == "v1" and second == "v2":
                    header_consumed = True
                    continue
                header_consumed = True

            if len(row) < 2:
                skipped += 1
                continue

            label = normalize_label(row[0])
            text = row[1].strip()

            if label is None or not text:
                skipped += 1
                continue

            payload = {
                "text": text,
                "label": label,
                "category": args.ham_category if label == 0 else args.spam_category,
            }

            key = text.lower()
            if label == 0:
                if key in seen_ham:
                    continue
                seen_ham.add(key)
                ham_f.write(json.dumps(payload, ensure_ascii=True) + "\n")
                ham_count += 1
            else:
                if key in seen_spam:
                    continue
                seen_spam.add(key)
                spam_f.write(json.dumps(payload, ensure_ascii=True) + "\n")
                spam_count += 1

    print(f"Input: {in_path}")
    print(f"Output ham: {ham_out} ({ham_count} rows)")
    print(f"Output spam: {spam_out} ({spam_count} rows)")
    print(f"Skipped malformed/empty rows: {skipped}")


if __name__ == "__main__":
    main()
