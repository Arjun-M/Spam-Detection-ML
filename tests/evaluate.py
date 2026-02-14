import argparse
import json
from collections import Counter
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from model import run_model


def load_suite(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
    if not rows:
        raise ValueError("Evaluation suite is empty")
    return rows


def evaluate(rows):
    y_true = []
    y_pred = []
    spam_true_cat = []
    spam_pred_cat = []
    errors = []

    for row in rows:
        text = row["text"]
        expected_label = int(row["label"])
        expected_category = str(row["category"])

        result = run_model(text)
        if not result["ok"]:
            raise RuntimeError(f"Invalid eval text {text!r}: {result['error']}")

        pred = result["result"]
        pred_label = 1 if pred["is_spam"] else 0
        pred_category = pred["category"]

        y_true.append(expected_label)
        y_pred.append(pred_label)

        if expected_label == 1:
            spam_true_cat.append(expected_category)
            spam_pred_cat.append(pred_category)

        if pred_label != expected_label:
            errors.append(
                {
                    "text": text,
                    "expected_label": expected_label,
                    "predicted_label": pred_label,
                    "expected_category": expected_category,
                    "predicted_category": pred_category,
                    "confidence": pred["confidence"],
                    "threshold_used": pred.get("threshold_used"),
                }
            )

    metrics = {
        "samples": len(rows),
        "label_distribution": dict(Counter(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "spam_category_accuracy": accuracy_score(spam_true_cat, spam_pred_cat) if spam_true_cat else 0.0,
        "errors": errors,
        "spam_category_report": classification_report(
            spam_true_cat, spam_pred_cat, zero_division=0, output_dict=True
        ) if spam_true_cat else {},
    }
    return metrics


def print_report(metrics, max_errors):
    print("Evaluation Report")
    print(f"- Samples: {metrics['samples']}")
    print(f"- Label distribution: {metrics['label_distribution']}")
    print(f"- Accuracy: {metrics['accuracy']:.4f}")
    print(f"- Precision: {metrics['precision']:.4f}")
    print(f"- Recall: {metrics['recall']:.4f}")
    print(f"- F1: {metrics['f1']:.4f}")
    print(f"- Spam category accuracy: {metrics['spam_category_accuracy']:.4f}")
    print(f"- Confusion matrix [tn fp; fn tp]: {metrics['confusion_matrix']}")

    errors = metrics["errors"]
    print(f"- Errors: {len(errors)}")
    for err in errors[:max_errors]:
        print(
            "  "
            f"text={err['text']!r} expected={err['expected_label']} "
            f"pred={err['predicted_label']} conf={err['confidence']} "
            f"cat={err['predicted_category']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate spam model on fixed messaging suite.")
    parser.add_argument("--suite", default="dataset/eval_suite.jsonl", help="Path to eval suite JSONL")
    parser.add_argument("--max-errors", type=int, default=30, help="Max misclassified samples to print")
    parser.add_argument("--min-precision", type=float, default=0.95)
    parser.add_argument("--min-recall", type=float, default=0.95)
    parser.add_argument("--min-f1", type=float, default=0.95)
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    args = parser.parse_args()

    rows = load_suite(args.suite)
    metrics = evaluate(rows)
    print_report(metrics, args.max_errors)

    failing = []
    if metrics["precision"] < args.min_precision:
        failing.append(f"precision {metrics['precision']:.4f} < {args.min_precision:.4f}")
    if metrics["recall"] < args.min_recall:
        failing.append(f"recall {metrics['recall']:.4f} < {args.min_recall:.4f}")
    if metrics["f1"] < args.min_f1:
        failing.append(f"f1 {metrics['f1']:.4f} < {args.min_f1:.4f}")
    if metrics["accuracy"] < args.min_accuracy:
        failing.append(f"accuracy {metrics['accuracy']:.4f} < {args.min_accuracy:.4f}")

    if failing:
        print("\nQUALITY GATE: FAIL")
        for line in failing:
            print(f"- {line}")
        raise SystemExit(1)

    print("\nQUALITY GATE: PASS")


if __name__ == "__main__":
    main()
