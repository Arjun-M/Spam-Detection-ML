import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

EXPECTED_KEYS = {"text", "label", "category"}
MAIN_CATEGORIES = {
    "adult",
    "crypto",
    "giveaway",
    "job_scam",
    "marketing",
    "normal",
    "phishing",
}


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def lint_dataset(data_dir: str = "dataset") -> dict:
    data_path = Path(data_dir)
    files = sorted(p for p in data_path.glob("*.jsonl") if p.name != "eval_suite.jsonl")

    errors = []
    warnings = []
    counts_by_category = Counter()
    counts_by_label = Counter()
    file_counts = {}

    text_occurrences = defaultdict(list)

    for fp in files:
        file_total = 0
        with fp.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                raw = line.strip()
                if not raw:
                    warnings.append(f"{fp}:{line_no} empty line")
                    continue

                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    errors.append(f"{fp}:{line_no} invalid json: {exc}")
                    continue

                file_total += 1

                keys = set(row.keys())
                if keys != EXPECTED_KEYS:
                    errors.append(
                        f"{fp}:{line_no} expected keys {sorted(EXPECTED_KEYS)}, got {sorted(keys)}"
                    )
                    continue

                text = row.get("text")
                label = row.get("label")
                category = row.get("category")

                if not isinstance(text, str) or not text.strip():
                    errors.append(f"{fp}:{line_no} invalid text")
                    continue
                if not isinstance(label, int) or label not in (0, 1):
                    errors.append(f"{fp}:{line_no} invalid label {label!r}")
                    continue
                if not isinstance(category, str) or not category.strip():
                    errors.append(f"{fp}:{line_no} invalid category {category!r}")
                    continue

                category = category.strip()
                if category == "normal" and label != 0:
                    errors.append(f"{fp}:{line_no} category=normal must have label=0")
                if category != "normal" and label != 1:
                    errors.append(
                        f"{fp}:{line_no} category={category} should usually have label=1"
                    )

                norm = normalize_text(text)
                text_occurrences[norm].append((str(fp), line_no, label, category))

                word_count = len(norm.split())
                if word_count < 2:
                    warnings.append(f"{fp}:{line_no} very short text ({word_count} words)")
                if word_count > 80:
                    warnings.append(f"{fp}:{line_no} very long text ({word_count} words)")

                counts_by_category[category] += 1
                counts_by_label[label] += 1

        file_counts[str(fp)] = file_total

    # Conflicting labels/categories for exact same normalized text.
    for norm_text, refs in text_occurrences.items():
        combos = {(label, cat) for (_, _, label, cat) in refs}
        if len(combos) > 1:
            locations = ", ".join(f"{fp}:{ln}" for (fp, ln, _, _) in refs[:6])
            errors.append(
                f"conflicting labels/categories for text={norm_text!r} at {locations}"
            )

    # Main file category checks.
    for cat in MAIN_CATEGORIES:
        fp = data_path / f"{cat}.jsonl"
        if not fp.exists():
            errors.append(f"missing main dataset file: {fp}")
            continue

    report = {
        "files_scanned": len(files),
        "file_counts": dict(file_counts),
        "counts_by_category": dict(counts_by_category),
        "counts_by_label": dict(counts_by_label),
        "errors": errors,
        "warnings": warnings,
        "ok": len(errors) == 0,
    }
    return report


def print_report(report: dict) -> None:
    print("Data Quality Lint")
    print(f"- Files scanned: {report['files_scanned']}")
    print(f"- Category counts: {report['counts_by_category']}")
    print(f"- Label counts: {report['counts_by_label']}")
    print(f"- Errors: {len(report['errors'])}")
    print(f"- Warnings: {len(report['warnings'])}")

    for err in report["errors"][:30]:
        print(f"  ERROR: {err}")
    if len(report["errors"]) > 30:
        print(f"  ... {len(report['errors']) - 30} more errors")

    for warn in report["warnings"][:20]:
        print(f"  WARN: {warn}")
    if len(report["warnings"]) > 20:
        print(f"  ... {len(report['warnings']) - 20} more warnings")


def main():
    parser = argparse.ArgumentParser(description="Lint dataset JSONL files for quality issues")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--strict-warnings", action="store_true")
    args = parser.parse_args()

    report = lint_dataset(args.data_dir)
    print_report(report)

    if not report["ok"]:
        raise SystemExit(1)
    if args.strict_warnings and report["warnings"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
