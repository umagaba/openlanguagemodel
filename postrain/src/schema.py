"""
Every raw dataset gets mapped into ONE common record shape before anything
else happens to it. This is what makes dedup/cleaning/splitting/formatting
generic instead of per-dataset special-cased at every stage.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
import hashlib

LABELS = [
    "hate_speech",
    "sexual_harassment",
    "explicit_material",
    "mental_health_risk",
    "legal_ambiguity",
    "racial_profiling",
    "none",
]


@dataclass
class UnifiedExample:
    text: str                                  # the raw text to be classified
    labels: Dict[str, int]                     # {label_name: 0/1}, all LABELS present
    source_id: str                             # e.g. "hatexplain", "jigsaw_civil_comments"
    source_license: str                        # copied from manifest.yaml at normalize time
    origin_id: Optional[str] = None            # original row/example id in the source, for traceability
    lang: Optional[str] = None                 # filled in during cleaning stage
    text_hash: Optional[str] = None            # filled in during cleaning stage (dedup key)
    split: Optional[str] = None                # "train" / "val" / "test", filled in at split stage

    def __post_init__(self):
        missing = [l for l in LABELS if l not in self.labels]
        for l in missing:
            self.labels[l] = 0

    def compute_hash(self) -> str:
        norm = " ".join(self.text.lower().split())
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def to_json(self) -> dict:
        return asdict(self)
