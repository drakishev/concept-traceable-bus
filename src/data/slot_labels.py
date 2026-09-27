"""Shared slot extraction + label encoding for VL-JEPA auxiliary heads.

Converts clinical concepts to integer class indices for cross-entropy
supervision. Missing/unknown labels map to -100 so PyTorch's CrossEntropyLoss
skips them.

Descriptor labels come from BUS-CoT's own structured fields
(`DatasetFiles/lesion_dataset.json`) rather than from regexes over the report,
for two reasons discovered while preparing the descriptor-concept experiments:

1. **Coverage.** Paraphrase augmentation renders each report in four surface
   styles; the legacy regexes matched only style A, so descriptor labels covered
   just 17.5% of the augmented training set. Keying on the image path instead
   recovers the label for every augmented copy of a study (~100%).
2. **Consistency.** `build_augmented.py` renders one source enum as several
   interchangeable synonyms (`Regular` -> "circumscribed" / "well-defined" /
   "smooth"). The legacy `MARGINS_CLASSES` treated those synonyms as distinct
   classes, so the head was asked to separate different phrasings of the *same*
   finding. Margins are therefore now labelled by the source enum (3 classes).

Shape and orientation have no structured enum; they exist only in the source
`reasoning_response`, so they are regexed out of that field (not out of the
possibly-paraphrased training text), which also gives full coverage.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

# ── Class vocabularies ────────────────────────────────────────────────────

PATHOLOGY_CLASSES = {"benign": 0, "malignant": 1}                     # 2-way
ORIENTATION_CLASSES = {"parallel": 0, "not parallel": 1}              # 2-way
RISK_CLASSES = {"low": 0, "high": 1}                                  # 2-way

# BI-RADS as a 7-way head: 2, 3, 4A, 4B, 4C, 5, 6
BIRADS_CLASSES = {"2": 0, "3": 1, "4A": 2, "4B": 3, "4C": 4, "5": 5, "6": 6}

# Descriptor vocabularies == the source enums in lesion_dataset.json.
MARGINS_CLASSES = {"regular": 0, "partiallyregular": 1, "irregular": 2}
ECHO_CLASSES = {
    "lowecho": 0, "slightlylowecho": 1, "isoechoic": 2, "highecho": 3,
    "heterogeneousecho": 4, "cysticsolidmixedecho": 5, "noecho": 6,
}
BOUNDARY_CLASSES = {
    "boundaryclear": 0, "boundaryfairlyclear": 1,
    "boundarysomewhatunclear": 2, "boundaryunclear": 3,
}
CALCIFICATION_CLASSES = {
    "nocalcification": 0, "microcalcification": 1, "multiplecalcifications": 2,
    "multipleclusteredmicrocalcifications": 3, "coarsecalcification": 4,
    "suspectedcalcification": 5,
}

SHAPE_CLASSES = {"oval": 0, "round": 1, "irregular": 2}

# ── Surface vocabularies ──────────────────────────────────────────────────
# Source enum -> the natural-language variants used by the paraphrase styles
# (scripts/preprocess/build_augmented.py; index selects the synonym per style).
# The evaluation extractor inverts these maps, so a report in any training style
# is read back to the same enum.

ECHO_MAP = {
    "LowEcho": ("hypoechoic", "low echogenicity", "markedly hypoechoic"),
    "SlightlyLowEcho": ("slightly hypoechoic", "mildly hypoechoic", "subtly hypoechoic"),
    "Isoechoic": ("isoechoic", "iso-echoic", "similar echogenicity to surrounding tissue"),
    "HighEcho": ("hyperechoic", "high echogenicity", "echogenic"),
    "HeterogeneousEcho": ("heterogeneous echo", "mixed echogenicity", "heterogeneous echotexture"),
    "CysticSolidMixedEcho": ("mixed cystic-solid", "complex echotexture",
                             "cystic and solid components"),
    "NoEcho": ("anechoic", "no internal echoes", "completely anechoic"),
}

CALC_MAP = {
    "NoCalcification": ("no calcifications", "absence of calcific deposits", "no calcific foci"),
    "Microcalcification": ("microcalcifications", "fine calcific deposits",
                           "scattered microcalcifications"),
    "MultipleCalcifications": ("multiple calcifications", "numerous calcific deposits",
                               "multifocal calcifications"),
    "MultipleClusteredMicrocalcifications": ("clustered microcalcifications",
                                             "grouped microcalcific foci",
                                             "clustered calcific particles"),
    "CoarseCalcification": ("coarse calcifications", "large calcific deposits",
                            "coarse calcific foci"),
    "SuspectedCalcification": ("suspected calcifications", "possible calcific deposits",
                               "equivocal calcifications"),
}

EDGE_MAP = {
    "Regular": ("circumscribed", "well-defined", "smooth"),
    "PartiallyRegular": ("partially circumscribed", "partially defined", "microlobulated"),
    "Irregular": ("irregular", "spiculated", "poorly defined"),
}

BOUNDARY_MAP = {
    "BoundaryClear": ("clear boundary", "well-circumscribed", "sharply demarcated"),
    "BoundaryFairlyClear": ("fairly clear boundary", "mostly well-defined margins",
                            "partially circumscribed"),
    "BoundarySomewhatUnclear": ("somewhat unclear boundary", "indistinct margins",
                                "partially ill-defined"),
    "BoundaryUnclear": ("unclear boundary", "ill-defined margins", "indistinct border"),
}

#: Words the BUS-CoT source reasoning texts (style A) use for the same enums.
#: They are not in the paraphrase maps, so the extractor needs them separately.
STYLE_A_ECHO = {
    "hypoechoic": "LowEcho", "slightly hypoechoic": "SlightlyLowEcho",
    "isoechoic": "Isoechoic", "hyperechoic": "HighEcho", "heterogeneous": "HeterogeneousEcho",
    "complex cystic and solid": "CysticSolidMixedEcho", "anechoic": "NoEcho",
}
STYLE_A_CALC = {
    "absence of calcific deposits": "NoCalcification",
    "punctuate microcalcifications": "Microcalcification",
    "multiple scattered calcifications": "MultipleCalcifications",
    "clustered microcalcific clusters": "MultipleClusteredMicrocalcifications",
    "coarse heterogeneous calcifications": "CoarseCalcification",
    "equivocal calcific particles": "SuspectedCalcification",
}

#: Legacy surface-form margin vocabulary, kept only so that the synonyms emitted
#: by the four paraphrase styles can be folded back onto the source enum when a
#: record cannot be matched to lesion_dataset.json.
_MARGIN_SYNONYM_TO_ENUM = {
    "circumscribed": "regular", "well-defined": "regular", "smooth": "regular",
    "partially circumscribed": "partiallyregular", "partially defined": "partiallyregular",
    "microlobulated": "partiallyregular", "angular": "partiallyregular",
    "irregular": "irregular", "spiculated": "irregular",
    "poorly defined": "irregular", "indistinct": "irregular",
}

LOW_RISK_BIRADS = {"2", "3"}

IGNORE_INDEX = -100

# ── Structured-field lookup (BUS-CoT) ─────────────────────────────────────

LESION_JSON = Path("data/raw/bus_cot/BUSCoT/DatasetFiles/lesion_dataset.json")
#: ".../BUS-Lesion/trainval/001755@0.png" -> "trainval_001755@0.png"
_KEY_RE = re.compile(r"BUS-Lesion/(trainval|test)/([^/]+)$")
_STRUCT: dict[str, dict] | None = None


def lesion_key(image_path: str | None) -> str | None:
    m = _KEY_RE.search(image_path or "")
    return f"{m.group(1)}_{m.group(2)}" if m else None


def _struct_lookup() -> dict[str, dict]:
    """Lazily load lesion_dataset.json. Missing file -> empty (regex fallback)."""
    global _STRUCT
    if _STRUCT is None:
        try:
            _STRUCT = json.loads(LESION_JSON.read_text())
        except Exception:
            _STRUCT = {}
    return _STRUCT


def _norm(v) -> str | None:
    """Enum value -> lookup key (lowercased, spaces stripped)."""
    if v is None:
        return None
    s = str(v).strip().lower().replace(" ", "")
    return s or None


# ── Regex extractors ──────────────────────────────────────────────────────

_ORI_RE = re.compile(r"This lesion is (not parallel|parallel)", re.IGNORECASE)
_MAR_RE = re.compile(r"has ([\w\s]+?) margins", re.IGNORECASE)
_SHA_RE = re.compile(r"margins and ([\w]+) shape", re.IGNORECASE)
_BIR_RE = re.compile(r"BIRADS\s+(\w+)", re.IGNORECASE)
_PAT_RE = re.compile(r"(malignant|benign) lesion", re.IGNORECASE)


def _extract(pattern: re.Pattern, text: str) -> str | None:
    m = pattern.search(text)
    return m.group(1).strip() if m else None


def extract_slot_strings(text: str) -> dict[str, str | None]:
    """Return raw string values for each slot (or None)."""
    return {
        "orientation": _extract(_ORI_RE, text),
        "margins":     _extract(_MAR_RE, text),
        "shape":       _extract(_SHA_RE, text),
        "birads":      _extract(_BIR_RE, text),
        "pathology":   _extract(_PAT_RE, text),
    }


def encode_labels(text: str, metadata: dict | None = None,
                  image_path: str | None = None) -> dict[str, int]:
    """Convert a record to integer class indices for every auxiliary head.

    Priority per slot: structured source fields (when `image_path` resolves to a
    BUS-CoT lesion record) > metadata > regex over `text`. Missing/out-of-vocab
    values map to IGNORE_INDEX so CE loss skips them.

    Returns: pathology, risk, birads, orientation, margins, shape,
    echogenicity, boundary, calcification.
    """
    metadata = metadata or {}
    out: dict[str, int] = {}
    entry = _struct_lookup().get(lesion_key(image_path) or "") or {}
    us = entry.get("us_report") or {}

    # ── pathology — metadata is authoritative ─────────────────────────────
    p_meta = metadata.get("pathology")
    if p_meta and isinstance(p_meta, str):
        out["pathology"] = PATHOLOGY_CLASSES.get(p_meta.lower(), IGNORE_INDEX)
    else:
        slots = extract_slot_strings(text)
        p = slots.get("pathology")
        out["pathology"] = PATHOLOGY_CLASSES.get(p.lower(), IGNORE_INDEX) if p else IGNORE_INDEX

    # ── BI-RADS: structured source > metadata > regex over the report ─────
    # BUS-CoT records carry no `birads` in metadata, so before this the label
    # came from a regex that only matched one of the four paraphrase styles
    # (25% coverage). Reading us_report.BIRADS lifts it to ~100%.
    b_str: str | None = None
    b_src = us.get("BIRADS")
    b_meta = metadata.get("birads")
    if b_src is not None and str(b_src).strip().lower() not in ("", "none"):
        b_str = str(b_src).strip().upper()
    elif b_meta is not None:
        b_str = str(b_meta).upper()
    else:
        b = extract_slot_strings(text).get("birads")
        b_str = b.upper() if b else None
    if b_str:
        out["birads"] = BIRADS_CLASSES.get(b_str, IGNORE_INDEX)
        out["risk"]   = RISK_CLASSES["low" if b_str in LOW_RISK_BIRADS else "high"]
    else:
        out["birads"] = IGNORE_INDEX
        out["risk"]   = IGNORE_INDEX

    # ── descriptors: structured enums first, else fold text synonyms back ──
    for head, field, vocab in (
        ("margins",       "LesionEdge",                  MARGINS_CLASSES),
        ("echogenicity",  "EchoCharacteristics",         ECHO_CLASSES),
        ("boundary",      "LesionBoundary",              BOUNDARY_CLASSES),
        ("calcification", "LesionCalcificationFeatures", CALCIFICATION_CLASSES),
    ):
        out[head] = vocab.get(_norm(us.get(field)), IGNORE_INDEX)

    # Read every text-derived slot from the SOURCE reasoning text when we have
    # it, so all four paraphrase variants of a study get identical labels.
    src_text = str(us.get("reasoning_response") or "") or text
    slots = extract_slot_strings(src_text)

    # A handful of source records have LesionEdge=None; recover those from the
    # reasoning text by folding its surface synonym back onto the enum.
    if out["margins"] == IGNORE_INDEX:
        m = slots.get("margins")
        enum = _MARGIN_SYNONYM_TO_ENUM.get(m.lower().strip()) if m else None
        out["margins"] = MARGINS_CLASSES.get(enum, IGNORE_INDEX)

    # ── shape / orientation: no structured enum exists for these ───────────

    o = slots.get("orientation")
    out["orientation"] = ORIENTATION_CLASSES.get(o.lower(), IGNORE_INDEX) if o else IGNORE_INDEX

    s = slots.get("shape")
    out["shape"] = SHAPE_CLASSES.get(s.lower(), IGNORE_INDEX) if s else IGNORE_INDEX

    return out


# ── Class counts (for nn.Linear out_dim) ──────────────────────────────────

NUM_CLASSES = {
    "pathology": 2,
    "orientation": 2,
    "risk": 2,
    "birads": 7,
    "margins": 3,
    "shape": 3,
    "echogenicity": 7,
    "boundary": 4,
    "calcification": 6,
}
