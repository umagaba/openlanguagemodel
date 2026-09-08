"""
Regex-based first pass. For anything sourced from real personal disclosures
(Safecity reports, ConvAbuse logs, any future mental-health text), run this
BEFORE the text ever reaches a training file, and prefer also running
Presidio (see requirements.txt) for a second, higher-recall pass on
higher-stakes categories (sexual_harassment, mental_health_risk).
"""
import re

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?<!\d)(\+?\d[\d\-\s()]{7,}\d)(?!\d)")
SSN_LIKE_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
HANDLE_RE = re.compile(r"@\w{2,30}\b")


def scrub(text: str) -> str:
    t = EMAIL_RE.sub("[EMAIL]", text)
    t = SSN_LIKE_RE.sub("[ID_NUMBER]", t)
    t = PHONE_RE.sub("[PHONE]", t)
    t = HANDLE_RE.sub("[USERNAME]", t)
    return t


def run_presidio(text: str) -> str:
    """
    Optional stronger pass. Import lazily so the base pipeline doesn't hard-
    depend on presidio if you haven't installed it yet.
    """
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    results = analyzer.analyze(text=text, language="en")
    return anonymizer.anonymize(text=text, analyzer_results=results).text
