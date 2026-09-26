"""Shared clinical-slot extraction for BUS-CoT report predictions.

Single source of truth for the slot extractor used by the slot-accuracy
evaluation (`scripts/evaluate_slots.py`), the statistical analysis
(`scripts/stats.py`), the intervention analyses and the external evaluations.
Also provides per-sample binary label derivation for the two headline clinical
endpoints (malignancy, BI-RADS risk group).

The extractor reads every surface style the models are trained on: the BUS-CoT
source template (A) and the three paraphrase styles of
scripts/preprocess/build_augmented.py (B clinical note, C prose, D brief).
Descriptor values are returned as the SOURCE ENUM (lower-cased key of
src/data/slot_labels.py, e.g. "lowecho", "regular"), so a report in any style
compares equal to a reference in any other style; the surface words are kept in
the `*_text` keys. Revision 2 replaced a style-A-only extractor after the
retrained models started emitting styles B and C, which the old regexes read as
"unextractable" and which the coverage guard would then have punished as
degenerate output.
"""
from __future__ import annotations

import re

from src.data.slot_labels import (
    _MARGIN_SYNONYM_TO_ENUM,
    CALC_MAP,
    ECHO_MAP,
    STYLE_A_CALC,
    STYLE_A_ECHO,
)

LOW_RISK_BIRADS = {"2", "3"}


def _inverse(mapping: dict[str, tuple[str, ...]], extra: dict[str, str]) -> dict[str, str]:
    inv = {syn.lower(): enum.lower() for enum, syns in mapping.items() for syn in syns}
    inv.update({k.lower(): v.lower() for k, v in extra.items()})
    return inv


#: surface phrase -> enum key, longest phrase first so "slightly hypoechoic"
#: wins over "hypoechoic" and "coarse heterogeneous calcifications" over
#: "heterogeneous".
_ECHO_SYN = _inverse(ECHO_MAP, STYLE_A_ECHO)
_CALC_SYN = _inverse(CALC_MAP, STYLE_A_CALC)
_ECHO_PATTERNS = [(p, re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE))
                  for p in sorted(_ECHO_SYN, key=len, reverse=True)]
_CALC_PATTERNS = [(p, re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE))
                  for p in sorted(_CALC_SYN, key=len, reverse=True)]

_BIRADS_RE = re.compile(r"BI-?RADS(?:\s+classification\s+is)?:?\s*(\d[ABC]?)\b", re.IGNORECASE)
_PATHOLOGY_RE = re.compile(r"\b(benign|malignant)\b", re.IGNORECASE)
_ORIENTATION_RE = re.compile(r"\b(not parallel|parallel)\b", re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>\s*([01])\s*</answer>", re.IGNORECASE)
_HISTOPATH_RE = re.compile(r"histopathology category:\s*([^.]+)", re.IGNORECASE)
_SHAPE_RE = re.compile(r"Shape:\s*(\w+)|\b(\w+)\s+shape\b|with\s+(\w+)\s+morphology",
                       re.IGNORECASE)
_MARGINS_RE = re.compile(
    r"Margins:\s*([a-z][a-z -]*?)\.|(?:^|[.,]\s*)([a-z-]+(?: [a-z-]+)?)\s+margins\b",
    re.IGNORECASE)
#: Style A glues the echo word to "with" ("hypoechoicwith absence of ...").
_GLUED_WITH_RE = re.compile(
    r"(hypoechoic|isoechoic|hyperechoic|anechoic|heterogeneous|solid)with\b", re.IGNORECASE)


def _first_group(m: re.Match | None) -> str | None:
    if m is None:
        return None
    return next((g.strip() for g in m.groups() if g), None)


def _margins_enum(phrase: str | None) -> str | None:
    """Fold a margin phrase onto the source enum; a two-word capture such as
    "has microlobulated" falls back to its last word."""
    if not phrase:
        return None
    words = phrase.lower().split()
    for candidate in (" ".join(words), words[-1]):
        if candidate in _MARGIN_SYNONYM_TO_ENUM:
            return _MARGIN_SYNONYM_TO_ENUM[candidate]
    return None


def _phrase(text: str, patterns: list) -> tuple[str | None, re.Match | None]:
    for phrase, rx in patterns:
        m = rx.search(text)
        if m:
            return phrase, m
    return None, None


def detect_style(text: str) -> str:
    """Which training template the text follows (A source, B note, C prose, D brief)."""
    if "Orientation:" in text or "Margins:" in text or "Impression:" in text:
        return "B"
    if "oriented" in text or "morphology" in text or "Overall assessment" in text:
        return "C"
    if "Ultrasound findings" in text:
        return "D"
    if "This lesion" in text or "BIRADS" in text:
        return "A"
    return "unknown"


def extract_slots(text: str) -> dict[str, str | None]:
    """Parse the clinical fields out of a report in any of the four training styles.

    Returns enum-level values for descriptors (`margins`, `echogenicity`,
    `calcification`), the surface words in `*_text`, and the raw strings for
    `orientation`, `shape`, `birads`, `pathology`, `answer`, `histopath`.
    """
    text = _GLUED_WITH_RE.sub(r"\1 with", text)
    calc_text, calc_match = _phrase(text, _CALC_PATTERNS)
    # the calcification phrase can contain an echo word ("coarse heterogeneous
    # calcifications"), so echo is read from the text with that phrase removed
    echo_source = text[:calc_match.start()] + text[calc_match.end():] if calc_match else text
    echo_text, _ = _phrase(echo_source, _ECHO_PATTERNS)
    margins_text = _first_group(_MARGINS_RE.search(text))
    if margins_text and margins_text.lower().startswith("has "):
        margins_text = margins_text[4:]  # style A: ", has circumscribed margins"
    birads = _first_group(_BIRADS_RE.search(text))
    return {
        "style": detect_style(text),
        "orientation": (_first_group(_ORIENTATION_RE.search(text)) or "").lower() or None,
        "margins": _margins_enum(margins_text),
        "margins_text": margins_text,
        "shape": (_first_group(_SHAPE_RE.search(text)) or "").lower() or None,
        "echogenicity": _ECHO_SYN.get(echo_text) if echo_text else None,
        "echogenicity_text": echo_text,
        "calcification": _CALC_SYN.get(calc_text) if calc_text else None,
        "calcification_text": calc_text,
        "birads": birads.upper() if birads else None,
        "histopath": _first_group(_HISTOPATH_RE.search(text)),
        "pathology": (_first_group(_PATHOLOGY_RE.search(text)) or "").lower() or None,
        "answer": _first_group(_ANSWER_RE.search(text)),
    }


#: Descriptor slots that a faithful intervention should co-vary, as opposed to
#: the assessment slots the intervention directly forces. Enum-level, so a
#: change of wording within one enum does not count as a change.
DESCRIPTOR_SLOTS = ("orientation", "margins", "shape", "echogenicity", "calcification")


def pathology_label(slots: dict[str, str | None]) -> int | None:
    """1 = malignant, 0 = benign, None = not extractable."""
    v = slots.get("pathology")
    if v is None:
        return None
    return 1 if v.lower() == "malignant" else 0


def risk_label(slots: dict[str, str | None]) -> int | None:
    """1 = high risk (BI-RADS 4A-6), 0 = low risk (2-3), None = not extractable."""
    v = slots.get("birads")
    if v is None:
        return None
    return 0 if v.upper() in LOW_RISK_BIRADS else 1


def paired_labels(predictions: list[dict], endpoint: str,
                  penalize_missing: bool = False) -> list[tuple[int, int]]:
    """Return [(ref_label, pred_label), ...] over samples where the REFERENCE label
    is extractable. endpoint: "pathology" or "risk".

    penalize_missing=False (default) reproduces `evaluate_slots.binary_metrics`
    (skip samples whose prediction is unextractable) — use for CIs that match the
    reported table numbers. penalize_missing=True counts an unextractable
    prediction as wrong (pred = 1-ref), which is the honest view that exposes
    degenerate outputs (e.g. the MedGemma-4B-VLM spurious 1.0).
    """
    fn = pathology_label if endpoint == "pathology" else risk_label
    out: list[tuple[int, int]] = []
    for d in predictions:
        rl = fn(extract_slots(d["reference"]))
        if rl is None:
            continue
        pl = fn(extract_slots(d["prediction"]))
        if pl is None:
            if not penalize_missing:
                continue
            pl = 1 - rl
        out.append((rl, pl))
    return out


def coverage(predictions: list[dict], endpoint: str) -> float:
    """Fraction of reference-labelled samples whose PREDICTION is also extractable.
    Low coverage flags a format/degeneracy mismatch that inflates skip-based F1."""
    fn = pathology_label if endpoint == "pathology" else risk_label
    ref_n = pred_n = 0
    for d in predictions:
        if fn(extract_slots(d["reference"])) is None:
            continue
        ref_n += 1
        if fn(extract_slots(d["prediction"])) is not None:
            pred_n += 1
    return pred_n / ref_n if ref_n else 0.0


def binary_f1(pairs: list[tuple[int, int]]) -> float:
    tp = sum(1 for r, p in pairs if r == 1 and p == 1)
    fp = sum(1 for r, p in pairs if r == 0 and p == 1)
    fn = sum(1 for r, p in pairs if r == 1 and p == 0)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
