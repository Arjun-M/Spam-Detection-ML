import json
import os
import re
from urllib.parse import urlparse

import joblib
from scipy.sparse import hstack

import config
from utils import preprocess_text

# Global variables to hold the loaded models and vectorizer
_vectorizer = None
_binary_model = None
_category_model = None
_metadata = None

# Signals that are strong spam indicators even in short messages.
SPAM_HINT_PATTERN = re.compile(
    r"(http|www|free|win|winner|claim|click|offer|bonus|urgent|verify|password|"
    r"account|bank|deposit|earn|investment|crypto|btc|telegram|airdrop|giveaway|jackpot|prize)",
    re.IGNORECASE,
)

BENIGN_WIN_CONTEXT_PATTERN = re.compile(
    r"\b(won|win|winner)\b.*\b(match|game|tournament|league|race|finals|team|football|cricket|basketball)\b",
    re.IGNORECASE,
)

SCAM_ACTION_PATTERN = re.compile(
    r"(claim|click|prize|reward|link|http|www|free|money|cash|gift|airdrop|crypto|account|verify|urgent)",
    re.IGNORECASE,
)

GIVEAWAY_OVERRIDE_PATTERN = re.compile(
    r"(\b(won|winner|jackpot|lucky draw)\b.*\b(lambo|lamboo|prize|reward|gift|voucher|tesla|iphone|cash)\b)|"
    r"(\b(claim|redeem)\b.*\b(prize|reward|gift|voucher)\b)",
    re.IGNORECASE,
)

SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
URL_TOKEN_PATTERN = re.compile(r"^(https?://\S+|www\.\S+)$", re.IGNORECASE)
URL_ANY_PATTERN = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
LINK_SPAM_CUE_PATTERN = re.compile(
    r"(claim|verify|password|bank|urgent|winner|prize|reward|free|bonus|airdrop|crypto|"
    r"deposit|investment|gift card|limited time|act now|suspended|login|otp|kyc)",
    re.IGNORECASE,
)

BENIGN_ADULT_CONTEXT_PATTERN = re.compile(
    r"(older than 18|over 18|under 18|age requirement|adult supervision|age limit|"
    r"content rating|parental guidance|legal age|years old)",
    re.IGNORECASE,
)

BENIGN_WORK_CONTEXT_PATTERN = re.compile(
    r"(pull request|code review|deployment|sprint|bug fix|qa|release note|"
    r"project update|meeting notes|standup|ticket|merge request|ci pipeline)",
    re.IGNORECASE,
)


def _load_metadata():
    if os.path.exists(config.METADATA_PATH):
        with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "spam_threshold": config.SPAM_THRESHOLD,
        "short_text_word_count": config.SHORT_TEXT_WORD_COUNT,
        "short_text_threshold": config.SHORT_TEXT_THRESHOLD,
        "very_short_text_word_count": config.VERY_SHORT_TEXT_WORD_COUNT,
        "very_short_text_threshold": config.VERY_SHORT_TEXT_THRESHOLD,
    }


def load_models():
    """Loads the model and vectorizer from disk only once."""
    global _vectorizer, _binary_model, _category_model, _metadata

    if _vectorizer is None:
        _vectorizer = joblib.load(config.VECTORIZER_PATH)
    if _binary_model is None:
        _binary_model = joblib.load(config.BINARY_MODEL_PATH)
    if _category_model is None:
        _category_model = joblib.load(config.CATEGORY_MODEL_PATH)
    if _metadata is None:
        _metadata = _load_metadata()


def _vectorize_single(cleaned_text):
    # Backward compatible with old single-vectorizer format.
    if isinstance(_vectorizer, dict) and "word" in _vectorizer and "char" in _vectorizer:
        word_vec = _vectorizer["word"].transform([cleaned_text])
        char_vec = _vectorizer["char"].transform([cleaned_text])
        return hstack([word_vec, char_vec], format="csr")

    return _vectorizer.transform([cleaned_text])


def _effective_threshold(raw_text, cleaned_text):
    threshold = float(_metadata.get("spam_threshold", config.SPAM_THRESHOLD))
    short_word_count = int(
        _metadata.get("short_text_word_count", config.SHORT_TEXT_WORD_COUNT)
    )
    short_threshold = float(
        _metadata.get("short_text_threshold", config.SHORT_TEXT_THRESHOLD)
    )
    very_short_word_count = int(
        _metadata.get("very_short_text_word_count", config.VERY_SHORT_TEXT_WORD_COUNT)
    )
    very_short_threshold = float(
        _metadata.get("very_short_text_threshold", config.VERY_SHORT_TEXT_THRESHOLD)
    )

    words = [w for w in cleaned_text.split(" ") if w]
    has_spam_hint = bool(SPAM_HINT_PATTERN.search(raw_text or ""))

    # Short chat messages are frequent in messaging apps, and without spam hints
    # they need a stricter threshold to avoid false positives.
    if not has_spam_hint:
        if len(words) <= very_short_word_count:
            threshold = max(threshold, very_short_threshold)
        elif len(words) <= short_word_count:
            threshold = max(threshold, short_threshold)

    return threshold


def _is_benign_win_context(raw_text):
    if not raw_text:
        return False
    return bool(BENIGN_WIN_CONTEXT_PATTERN.search(raw_text)) and not bool(
        SCAM_ACTION_PATTERN.search(raw_text)
    )


def _is_benign_context(raw_text):
    if not raw_text:
        return False
    if bool(SCAM_ACTION_PATTERN.search(raw_text)):
        return False
    return bool(BENIGN_ADULT_CONTEXT_PATTERN.search(raw_text)) or bool(
        BENIGN_WORK_CONTEXT_PATTERN.search(raw_text)
    )




def _extract_single_url_domain(raw_text: str) -> str | None:
    if not raw_text:
        return None
    candidate = raw_text.strip()
    if not URL_TOKEN_PATTERN.match(candidate):
        return None
    if candidate.lower().startswith("www."):
        candidate = "https://" + candidate
    try:
        parsed = urlparse(candidate)
        host = (parsed.netloc or "").lower().strip()
    except Exception:
        return None
    if not host:
        return None
    if host.startswith("m."):
        host = host[2:]
    return host




def _extract_url_domains(raw_text: str) -> list[str]:
    if not raw_text:
        return []
    domains = []
    for m in URL_ANY_PATTERN.finditer(raw_text):
        url = m.group(0).strip()
        if url.lower().startswith("www."):
            url = "https://" + url
        try:
            parsed = urlparse(url)
            host = (parsed.netloc or "").lower().strip()
        except Exception:
            continue
        if not host:
            continue
        if host.startswith("m."):
            host = host[2:]
        domains.append(host)
    return domains


def _has_blocked_domain(raw_text: str) -> bool:
    blocked = set(getattr(config, "BLOCKED_URL_DOMAINS", set()))
    if not blocked:
        return False
    domains = _extract_url_domains(raw_text)
    if not domains:
        return False
    for host in domains:
        if host in blocked:
            return True
        # Also block subdomains of blocked root domains.
        for base in blocked:
            if host.endswith('.' + base):
                return True
    return False


def _contains_url(raw_text: str) -> bool:
    return bool(URL_ANY_PATTERN.search(raw_text or ""))


def _has_link_spam_cues(raw_text: str) -> bool:
    return bool(LINK_SPAM_CUE_PATTERN.search(raw_text or ""))


def _split_long_text(text: str) -> list[str]:
    max_words = int(getattr(config, "CHUNK_MAX_WORDS", 40))
    max_chunks = int(getattr(config, "MAX_CHUNKS", 24))

    parts = [p.strip() for p in SENTENCE_SPLIT_PATTERN.split(text or "") if p.strip()]
    chunks = []
    current = []
    current_words = 0

    for part in parts:
        words = part.split()
        if not words:
            continue
        if len(words) > max_words:
            # Hard-split very long sentence-like parts.
            for i in range(0, len(words), max_words):
                piece = " ".join(words[i : i + max_words]).strip()
                if piece:
                    chunks.append(piece)
                if len(chunks) >= max_chunks:
                    return chunks[:max_chunks]
            continue

        if current_words + len(words) > max_words and current:
            chunks.append(" ".join(current).strip())
            current = [part]
            current_words = len(words)
        else:
            current.append(part)
            current_words += len(words)

        if len(chunks) >= max_chunks:
            return chunks[:max_chunks]

    if current and len(chunks) < max_chunks:
        chunks.append(" ".join(current).strip())

    return chunks[:max_chunks]


def _predict_single(raw_text: str, cleaned_text: str) -> dict:
    # URL policy:
    # 1) Blocked domain => spam
    # 2) URL without explicit scam cues => normal (link sharing is common)
    if _contains_url(raw_text):
        if _has_blocked_domain(raw_text):
            return {
                "is_spam": True,
                "confidence": 0.99,
                "category": "spam",
                "threshold_used": float(_metadata.get("spam_threshold", config.SPAM_THRESHOLD)),
            }
        if not _has_link_spam_cues(raw_text):
            return {
                "is_spam": False,
                "confidence": 0.05,
                "category": "normal",
                "threshold_used": float(_metadata.get("spam_threshold", config.SPAM_THRESHOLD)),
            }

    vec = _vectorize_single(cleaned_text)

    probs = _binary_model.predict_proba(vec)[0]
    spam_prob = float(probs[1])

    threshold = _effective_threshold(raw_text, cleaned_text)
    is_spam = spam_prob >= threshold

    if is_spam and spam_prob < 0.92 and _is_benign_win_context(raw_text):
        is_spam = False
    if is_spam and spam_prob < 0.85 and _is_benign_context(raw_text):
        is_spam = False

    has_giveaway_override = bool(GIVEAWAY_OVERRIDE_PATTERN.search(raw_text or ""))
    if not is_spam and has_giveaway_override and not _is_benign_win_context(raw_text):
        is_spam = True

    if is_spam:
        if has_giveaway_override:
            category = "giveaway"
        else:
            category = _category_model.predict(vec)[0]
    else:
        category = "normal"

    return {
        "is_spam": bool(is_spam),
        "confidence": float(spam_prob),
        "category": str(category),
        "threshold_used": float(threshold),
    }


def validate_message(text: str) -> tuple[bool, str]:
    """Validates user input text before running inference."""
    if text is None:
        return False, "Input is required."

    if not isinstance(text, str):
        return False, "Input must be a string."

    normalized = text.strip()
    if not normalized:
        return False, "Input cannot be empty."

    if len(normalized) < 2:
        return False, "Input is too short (minimum 2 characters)."

    if not any(ch.isalnum() for ch in normalized):
        return False, "Input must contain letters or numbers."

    return True, ""


def predict_message(text: str) -> dict:
    """
    Predicts if a message is spam and its category.

    Returns:
    {
        "is_spam": bool,
        "confidence": float,
        "category": str
    }
    """
    load_models()

    cleaned_text = preprocess_text(text)
    word_count = len([w for w in cleaned_text.split(" ") if w])
    long_threshold = int(getattr(config, "LONG_TEXT_WORD_THRESHOLD", 80))

    if word_count <= long_threshold:
        pred = _predict_single(text, cleaned_text)
        return {
            "is_spam": pred["is_spam"],
            "confidence": round(pred["confidence"], 4),
            "category": pred["category"],
            "threshold_used": round(pred["threshold_used"], 4),
            "chunked": False,
        }

    chunks = _split_long_text(text)
    if not chunks:
        pred = _predict_single(text, cleaned_text)
        return {
            "is_spam": pred["is_spam"],
            "confidence": round(pred["confidence"], 4),
            "category": pred["category"],
            "threshold_used": round(pred["threshold_used"], 4),
            "chunked": False,
        }

    chunk_predictions = []
    for chunk in chunks:
        cp = _predict_single(chunk, preprocess_text(chunk))
        chunk_predictions.append(cp)

    highest = max(chunk_predictions, key=lambda x: x["confidence"])
    spam_chunks = [cp for cp in chunk_predictions if cp["is_spam"]]
    is_spam = len(spam_chunks) > 0

    if is_spam:
        representative = max(spam_chunks, key=lambda x: x["confidence"])
    else:
        representative = highest

    return {
        "is_spam": bool(is_spam),
        "confidence": round(float(highest["confidence"]), 4),
        "category": str(representative["category"] if is_spam else "normal"),
        "threshold_used": round(float(representative["threshold_used"]), 4),
        "chunked": True,
        "chunk_count": len(chunks),
    }


def run_model(text: str) -> dict:
    """Validates input and returns model output or a validation error payload."""
    ok, error = validate_message(text)
    if not ok:
        return {
            "ok": False,
            "error": error,
            "input": text,
        }

    prediction = predict_message(text)
    return {
        "ok": True,
        "input": text.strip(),
        "result": prediction,
    }


def update_model(text: str, label: int, category: str):
    """Stores feedback data for offline retraining."""
    if text is None:
        return

    os.makedirs("dataset", exist_ok=True)
    feedback_path = os.path.join("dataset", "feedback.jsonl")
    payload = {
        "text": text.strip(),
        "label": int(label),
        "category": str(category),
    }

    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def get_model_specs() -> dict:
    """Returns metadata/specs about model configuration and saved artifacts."""
    specs = {
        "model_dir": config.MODEL_DIR,
        "binary_model_path": config.BINARY_MODEL_PATH,
        "category_model_path": config.CATEGORY_MODEL_PATH,
        "vectorizer_path": config.VECTORIZER_PATH,
        "metadata_path": config.METADATA_PATH,
        "spam_threshold": config.SPAM_THRESHOLD,
        "max_features": config.MAX_FEATURES,
        "files": {
            "binary_model_exists": os.path.exists(config.BINARY_MODEL_PATH),
            "category_model_exists": os.path.exists(config.CATEGORY_MODEL_PATH),
            "vectorizer_exists": os.path.exists(config.VECTORIZER_PATH),
            "metadata_exists": os.path.exists(config.METADATA_PATH),
        },
    }

    try:
        load_models()
        specs["loaded"] = True
        specs["binary_classes"] = [int(c) for c in _binary_model.classes_]
        specs["category_classes"] = [str(c) for c in _category_model.classes_]
        specs["vectorizer_type"] = type(_vectorizer).__name__
        specs["runtime_threshold"] = _metadata.get("spam_threshold", config.SPAM_THRESHOLD)
    except Exception as exc:
        specs["loaded"] = False
        specs["load_error"] = str(exc)

    return specs


def print_model_specs() -> None:
    """Pretty-prints model specs for direct module execution."""
    specs = get_model_specs()
    print("Model Specs")
    print(f"- Model dir: {specs['model_dir']}")
    print(f"- Binary model: {specs['binary_model_path']}")
    print(f"- Category model: {specs['category_model_path']}")
    print(f"- Vectorizer: {specs['vectorizer_path']}")
    print(f"- Metadata: {specs['metadata_path']}")
    print(f"- Base threshold: {specs['spam_threshold']}")
    print(f"- Max features: {specs['max_features']}")
    print(
        "- Files exist: "
        f"binary={specs['files']['binary_model_exists']}, "
        f"category={specs['files']['category_model_exists']}, "
        f"vectorizer={specs['files']['vectorizer_exists']}, "
        f"metadata={specs['files']['metadata_exists']}"
    )

    if specs.get("loaded"):
        print("- Loaded: True")
        print(f"- Runtime threshold: {specs['runtime_threshold']}")
        print(f"- Binary classes: {specs['binary_classes']}")
        print(f"- Category classes: {specs['category_classes']}")
        print(f"- Vectorizer type: {specs['vectorizer_type']}")
    else:
        print("- Loaded: False")
        print(f"- Load error: {specs.get('load_error', 'unknown error')}")


if __name__ == "__main__":
    print_model_specs()
