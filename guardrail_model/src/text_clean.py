import re
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Ensure consistent language detection results
DetectorFactory.seed = 0

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
MULTI_WS_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{4,}")  # aaaaaa -> aaaa
HTML_TAG_RE = re.compile(r"<[^>]+>")


def basic_clean(text: str, strip_mentions: bool = True, strip_urls: bool = True) -> str:
    if not isinstance(text, str):
        return ""
    t = HTML_TAG_RE.sub(" ", text)
    if strip_urls:
        t = URL_RE.sub(" ", t)
    if strip_mentions:
        t = MENTION_RE.sub(" ", t)
    t = REPEATED_CHAR_RE.sub(r"\1\1\1\1", t)
    t = MULTI_WS_RE.sub(" ", t).strip()
    return t


def length_ok(text: str, min_chars: int = 3, max_chars: int = 4000) -> bool:
    n = len(text)
    return min_chars <= n <= max_chars


def symbol_ratio_ok(text: str, max_ratio: float = 0.3) -> bool:
    """Heuristic quality filter: reject text that's mostly punctuation/symbols/emoji."""
    if not text:
        return False
    symbol_chars = sum(1 for c in text if not (c.isalnum() or c.isspace()))
    return (symbol_chars / max(len(text), 1)) <= max_ratio


def is_boilerplate(text: str) -> bool:
    boiler_markers = [
        "click here", "subscribe now", "terms of service",
        "all rights reserved", "lorem ipsum",
    ]
    low = text.lower()
    return any(m in low for m in boiler_markers)


def is_target_language(text: str, target_lang: str = "en") -> bool:
    """Detects if the text matches the target language."""
    if not text or len(text.strip()) < 10:
        # text is too short for reliable detection; fallback to True to let other heuristics handle it
        return True 
    try:
        return detect(text) == target_lang
    except LangDetectException:
        # If it fails to detect (e.g., all numbers/symbols), reject it
        return False