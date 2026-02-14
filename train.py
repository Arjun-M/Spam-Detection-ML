import json
import os
from collections import Counter

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

import config
from tests.data_quality_lint import lint_dataset
from utils import preprocess_text


DATASET_DIR = "dataset"
EXCLUDED_JSONL_FILES = {"eval_suite.jsonl"}


def load_all_data(data_dir):
    records = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".jsonl"):
            continue
        if filename in EXCLUDED_JSONL_FILES:
            continue
        file_path = os.path.join(data_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                records.append(
                    {
                        "text": preprocess_text(data["text"]),
                        "label": int(data["label"]),
                        "category": str(data["category"]),
                    }
                )
    return records


def deduplicate_records(records):
    deduped = {}
    for rec in records:
        text = rec["text"]
        if not text:
            continue
        if text not in deduped:
            deduped[text] = rec
    return list(deduped.values())


def add_hard_negatives(records):
    """Adds short, common chat phrases to reduce false positives in messaging apps."""
    hard_negative_texts = [
        "lol",
        "ok",
        "okay",
        "hi",
        "hello",
        "yo",
        "new video",
        "new video uploaded",
        "join now",
        "on my way",
        "call me",
        "see you soon",
        "good morning",
        "good night",
        "lets talk later",
        "send me details",
        "are you free",
        "what time",
        "got it",
        "thanks",
        "thank you",
        "can i help",
        "how can i help you",
    ]

    augmented = list(records)
    existing = {r["text"] for r in records}
    for text in hard_negative_texts:
        cleaned = preprocess_text(text)
        if cleaned in existing:
            continue
        augmented.append({"text": cleaned, "label": 0, "category": "normal"})
    return augmented


def add_targeted_spam_positives(records):
    """Adds high-signal scam variants observed in real chat usage."""
    targeted_spam_texts = [
        "i have just won a lamboo",
        "i have just won a lambo",
        "i have just won a lamboo hurray join me now",
        "congrats you have won a lamboo claim now",
        "you won a lambo click here to claim prize",
        "you won big hurray join now to claim",
        "winner alert you have won now join us",
        "claim your lambo prize now",
        "you are selected winner join now",
        "jackpot winner click now to claim your reward",
        "you have just won huge prize join now",
        "lucky winner join now and claim your gift",
    ]

    augmented = list(records)
    existing = {r["text"] for r in records}
    for text in targeted_spam_texts:
        cleaned = preprocess_text(text)
        if cleaned in existing:
            continue
        augmented.append({"text": cleaned, "label": 1, "category": "giveaway"})
    return augmented


def add_targeted_normal_contexts(records):
    """Adds benign contexts that share words with scams (e.g., won/join)."""
    normal_context_texts = [
        "i won football match today hurray",
        "we won the game today",
        "i won the tournament finals",
        "our team won again",
        "we just won the league",
        "join now for team practice",
        "join now for the meeting",
        "join now if you are free",
        "join me now for lunch",
        "hurray we won the match",
        "i have just won the office raffle fairly",
        "he won the race yesterday",
    ]

    augmented = list(records)
    existing = {r["text"] for r in records}
    for text in normal_context_texts:
        cleaned = preprocess_text(text)
        if cleaned in existing:
            continue
        augmented.append({"text": cleaned, "label": 0, "category": "normal"})
    return augmented


def build_features(train_texts, val_texts):
    word_vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=config.MAX_FEATURES,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=max(config.MAX_FEATURES // 2, 5000),
        sublinear_tf=True,
    )

    X_train_word = word_vectorizer.fit_transform(train_texts)
    X_val_word = word_vectorizer.transform(val_texts)

    X_train_char = char_vectorizer.fit_transform(train_texts)
    X_val_char = char_vectorizer.transform(val_texts)

    X_train = hstack([X_train_word, X_train_char], format="csr")
    X_val = hstack([X_val_word, X_val_char], format="csr")

    vectorizer_bundle = {"word": word_vectorizer, "char": char_vectorizer}
    return X_train, X_val, vectorizer_bundle


def pick_threshold(y_true, y_prob, target_min_precision):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    best_threshold = config.SPAM_THRESHOLD
    best_score = -1.0

    for idx, threshold in enumerate(thresholds):
        p = precision[idx + 1]
        r = recall[idx + 1]
        if p < target_min_precision:
            continue
        if p + r == 0:
            continue
        f1 = 2 * p * r / (p + r)
        if f1 > best_score:
            best_score = f1
            best_threshold = float(threshold)

    if best_score >= 0:
        return round(best_threshold, 4)

    # Fallback: maximize F1 without precision constraint.
    f1_scores = []
    for idx, threshold in enumerate(thresholds):
        p = precision[idx + 1]
        r = recall[idx + 1]
        f1 = 0.0 if (p + r) == 0 else (2 * p * r / (p + r))
        f1_scores.append((f1, threshold))

    if not f1_scores:
        return config.SPAM_THRESHOLD

    return round(float(max(f1_scores, key=lambda x: x[0])[1]), 4)


def train_best_binary_model(X_train_vec, y_train_bin, X_val_vec, y_val_bin):
    candidates = [
        {"C": 0.5, "class_weight": "balanced"},
        {"C": 1.0, "class_weight": "balanced"},
        {"C": 2.0, "class_weight": "balanced"},
        {"C": 3.0, "class_weight": "balanced"},
    ]

    best = None
    for params in candidates:
        model = LogisticRegression(
            max_iter=2500,
            class_weight=params["class_weight"],
            C=params["C"],
            solver="liblinear",
            random_state=42,
        )
        model.fit(X_train_vec, y_train_bin)
        y_prob = model.predict_proba(X_val_vec)[:, 1]
        threshold = pick_threshold(
            y_true=np.array(y_val_bin),
            y_prob=y_prob,
            target_min_precision=config.TARGET_MIN_PRECISION,
        )
        y_pred = (y_prob >= threshold).astype(int)
        precision = precision_score(y_val_bin, y_pred, zero_division=0)
        recall = recall_score(y_val_bin, y_pred, zero_division=0)
        f1 = f1_score(y_val_bin, y_pred, zero_division=0)
        # Prioritize precision first, then F1.
        score = precision * 10.0 + f1

        if best is None or score > best["score"]:
            best = {
                "model": model,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "score": score,
                "params": params,
                "y_prob": y_prob,
            }

    return best


def train():
    print("Running data quality lint...")
    lint_report = lint_dataset(DATASET_DIR)
    print(
        f"Lint summary: errors={len(lint_report['errors'])}, "
        f"warnings={len(lint_report['warnings'])}"
    )
    if lint_report["errors"]:
        first_error = lint_report["errors"][0]
        raise ValueError(f"Dataset lint failed. First error: {first_error}")

    print(f"Loading datasets from {DATASET_DIR}/ directory...")
    records = load_all_data(DATASET_DIR)
    print(f"Original dataset size: {len(records)}")

    records = deduplicate_records(records)
    print(f"After deduplication: {len(records)}")

    records = add_hard_negatives(records)
    print(f"After hard-negative augmentation: {len(records)}")
    records = add_targeted_spam_positives(records)
    print(f"After targeted-spam augmentation: {len(records)}")
    records = add_targeted_normal_contexts(records)
    print(f"After targeted-normal augmentation: {len(records)}")

    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    categories = [r["category"] for r in records]

    print(f"Label distribution: {Counter(labels)}")
    print(f"Category distribution: {Counter(categories)}")

    X_train_texts, X_val_texts, y_train_bin, y_val_bin, y_train_cat, y_val_cat = train_test_split(
        texts,
        labels,
        categories,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    print(f"Training on {len(X_train_texts)} samples, validating on {len(X_val_texts)} samples.")

    print("Building word + character TF-IDF features...")
    X_train_vec, X_val_vec, vectorizer_bundle = build_features(X_train_texts, X_val_texts)

    print("Training binary classifier with model selection...")
    best_binary = train_best_binary_model(
        X_train_vec=X_train_vec,
        y_train_bin=y_train_bin,
        X_val_vec=X_val_vec,
        y_val_bin=y_val_bin,
    )
    binary_model = best_binary["model"]
    y_prob_bin = best_binary["y_prob"]
    tuned_threshold = best_binary["threshold"]
    y_pred_bin = (y_prob_bin >= tuned_threshold).astype(int)

    print("\n--- Binary Classification Results ---")
    print(f"Chosen params: {best_binary['params']}")
    print(f"Threshold: {tuned_threshold}")
    print(f"Accuracy:  {accuracy_score(y_val_bin, y_pred_bin):.4f}")
    print(f"Precision: {precision_score(y_val_bin, y_pred_bin):.4f}")
    print(f"Recall:    {recall_score(y_val_bin, y_pred_bin):.4f}")
    print(f"F1-score:  {f1_score(y_val_bin, y_pred_bin):.4f}")
    print("Confusion matrix [tn fp; fn tp]:")
    print(confusion_matrix(y_val_bin, y_pred_bin))

    print("\nTraining spam category classifier...")
    train_spam_mask = np.array(y_train_bin) == 1
    val_spam_mask = np.array(y_val_bin) == 1

    spam_X_train = X_train_vec[train_spam_mask]
    spam_y_train = np.array(y_train_cat)[train_spam_mask]

    category_model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    category_model.fit(spam_X_train, spam_y_train)

    if val_spam_mask.any():
        spam_X_val = X_val_vec[val_spam_mask]
        spam_y_val = np.array(y_val_cat)[val_spam_mask]
        y_pred_cat = category_model.predict(spam_X_val)
        print("\n--- Spam Category Results (spam-only validation set) ---")
        print(f"Accuracy: {accuracy_score(spam_y_val, y_pred_cat):.4f}")
        print(classification_report(spam_y_val, y_pred_cat, zero_division=0))

    print("Saving models and metadata...")
    joblib.dump(vectorizer_bundle, config.VECTORIZER_PATH)
    joblib.dump(binary_model, config.BINARY_MODEL_PATH)
    joblib.dump(category_model, config.CATEGORY_MODEL_PATH)

    metadata = {
        "spam_threshold": tuned_threshold,
        "short_text_word_count": config.SHORT_TEXT_WORD_COUNT,
        "short_text_threshold": config.SHORT_TEXT_THRESHOLD,
        "very_short_text_word_count": config.VERY_SHORT_TEXT_WORD_COUNT,
        "very_short_text_threshold": config.VERY_SHORT_TEXT_THRESHOLD,
        "target_min_precision": config.TARGET_MIN_PRECISION,
    }
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Models saved to {config.MODEL_DIR}/")


if __name__ == "__main__":
    train()
