"""Build paraphrase-augmented training data from BUS-CoT.

Each training record is re-rendered in 4 different surface styles using
the structured fields from lesion_dataset.json. This quadruples the
training set and forces the model to learn clinical semantics rather than
memorising a single template pattern.

The 4 styles:
  A — original template (kept as-is)
  B — clinical note (structured, label-value pairs)
  C — passive/descriptive prose
  D — brief impression with key findings

Only train.jsonl is augmented; val/test stay clean for honest evaluation.

Histopathology is deliberately absent from every style (revision 2, reviewer 1
item 9): it cannot be read from sonographic appearance, so it is not a target.

Usage:
    python scripts/preprocess/build_augmented.py \
        --unified_dir data/unified \
        --lesion_json data/raw/bus_cot/BUSCoT/DatasetFiles/lesion_dataset.json \
        --out_dir data/augmented
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.data.slot_labels import CALC_MAP, ECHO_MAP, EDGE_MAP, lesion_key  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── Vocabulary mappings (structured enum → natural language variants) ──────
# Defined once in src/data/slot_labels.py, shared with the evaluation extractor.


def _v(mapping: dict, key: str | None, idx: int = 0) -> str | None:
    """Natural-language variant of a source enum; idx selects the synonym.
    None when the field is absent, so the renderer can omit it."""
    if key and key in mapping:
        opts = mapping[key]
        return opts[idx % len(opts)]
    return None


# ── Slot extraction from reasoning_response ───────────────────────────────

def extract_slots(text: str) -> dict:
    def _re(pat):
        m = re.search(pat, text, re.IGNORECASE)
        return m.group(1).strip() if m else None
    return {
        "orientation": _re(r"This lesion is (not parallel|parallel)"),
        "margins":     _re(r"has ([\w\s]+?) margins"),
        "shape":       _re(r"margins and ([\w]+) shape"),
        # BUS-CoT writes "BIRADS 4A", the BUS-BRA template "BI-RADS classification is 4"
        "birads":      _re(r"BI-?RADS(?: classification is)?\s+(\d[ABC]?)\b"),
        "pathology":   _re(r"(malignant|benign) lesion"),
        "answer":      _re(r"<answer>\s*([01])\s*</answer>"),
    }


# ── Style renderers ────────────────────────────────────────────────────────
# A field that is not annotated for a record (BUS-BRA has no descriptors; a few
# BUS-CoT records lack an echo enum) is OMITTED, never rendered as "unknown":
# a literal "unknown" target teaches the decoder an abstention token that it
# then emits on out-of-distribution images, which destroys extraction coverage.

def _fields(slots: dict, us_report: dict, idx: int) -> dict[str, str | None]:
    return {
        "orientation": slots.get("orientation"),
        "margins": slots.get("margins") or _v(EDGE_MAP, us_report.get("LesionEdge"), idx),
        "shape": slots.get("shape"),
        "echo": _v(ECHO_MAP, us_report.get("EchoCharacteristics"), idx),
        "calc": _v(CALC_MAP, us_report.get("LesionCalcificationFeatures"), idx),
        "birads": slots.get("birads") or us_report.get("BIRADS"),
        "pathology": slots.get("pathology"),
        "answer": slots.get("answer") or "?",
    }


def style_B(slots: dict, us_report: dict) -> str:
    """Clinical note style: structured label-value pairs."""
    f = _fields(slots, us_report, idx=1)
    parts = [
        f"Orientation: {f['orientation']}." if f["orientation"] else None,
        f"Margins: {f['margins']}." if f["margins"] else None,
        f"Shape: {f['shape']}." if f["shape"] else None,
        f"Echogenicity: {f['echo']}." if f["echo"] else None,
        f"Calcification: {f['calc']}." if f["calc"] else None,
        f"BI-RADS: {f['birads']}." if f["birads"] else None,
        f"Impression: {f['pathology']} lesion." if f["pathology"] else None,
        f"<answer> {f['answer']} </answer>",
    ]
    return " ".join(x for x in parts if x)


def style_C(slots: dict, us_report: dict) -> str:
    """Passive/descriptive prose."""
    f = _fields(slots, us_report, idx=2)
    parts = []
    if f["orientation"]:
        parts.append(f"The lesion is oriented {f['orientation']} to the skin surface.")
    if f["margins"] and f["shape"]:
        parts.append(f"{f['margins'].capitalize()} margins are observed with "
                     f"{f['shape']} morphology.")
    elif f["margins"]:
        parts.append(f"{f['margins'].capitalize()} margins are observed.")
    if f["echo"] and f["calc"]:
        parts.append(f"The lesion demonstrates {f['echo']} characteristics and {f['calc']}.")
    elif f["echo"]:
        parts.append(f"The lesion demonstrates {f['echo']} characteristics.")
    elif f["calc"]:
        parts.append(f"The lesion demonstrates {f['calc']}.")
    if f["birads"]:
        parts.append(f"These sonographic features are consistent with BI-RADS {f['birads']}.")
    if f["pathology"]:
        parts.append(f"Overall assessment: {f['pathology']} lesion.")
    parts.append(f"<answer> {f['answer']} </answer>")
    return " ".join(parts)


def style_D(slots: dict, us_report: dict) -> str:
    """Brief clinical impression."""
    f = _fields(slots, us_report, idx=0)
    findings = [x for x in (
        f["orientation"],
        f"{f['margins']} margins" if f["margins"] else None,
        f"{f['shape']} shape" if f["shape"] else None,
        f["echo"],
        f["calc"],
    ) if x]
    parts = []
    if findings:
        parts.append("Ultrasound findings: " + ", ".join(findings) + ".")
    if f["birads"] and f["pathology"]:
        parts.append(f"BI-RADS {f['birads']} — {f['pathology']}.")
    elif f["birads"]:
        parts.append(f"BI-RADS {f['birads']}.")
    elif f["pathology"]:
        parts.append(f"Impression: {f['pathology']}.")
    parts.append(f"<answer> {f['answer']} </answer>")
    return " ".join(parts)


STYLE_FUNCS = [style_B, style_C, style_D]
STYLE_NAMES = ["clinical_note", "descriptive", "brief"]


# ── Structured-field lookup ───────────────────────────────────────────────

def build_lesion_lookup(lesion_json_path: Path) -> dict[str, dict]:
    """Map lesion_dataset.json key ("trainval_001755@0.png") -> us_report dict."""
    with open(lesion_json_path) as f:
        raw = json.load(f)
    return {k: v["us_report"] for k, v in raw.items() if isinstance(v.get("us_report"), dict)}


def _strip_tags(text: str) -> str:
    """Strip <reasoning>...</reasoning> and <answer>...</answer> wrappers."""
    text = re.sub(r"<reasoning>(.*?)</reasoning>", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"<answer>.*?</answer>", "", text, flags=re.DOTALL)
    return text.strip()


def build_filename_lookup(lesion_json_path: Path) -> dict[str, dict]:
    """Build lookup from stripped reasoning text → us_report."""
    with open(lesion_json_path) as f:
        raw = json.load(f)
    response_to_report: dict[str, dict] = {}
    for entry in raw.values():
        rep = entry.get("us_report")
        if not isinstance(rep, dict):
            continue
        rr = rep.get("reasoning_response", "")
        if rr:
            response_to_report[rr.strip()] = rep
    return response_to_report


# ── Main ───────────────────────────────────────────────────────────────────

def augment_records(
    records: list[dict],
    lesion_lookup: dict[str, dict],
    response_to_report: dict[str, dict],
) -> list[dict]:
    """For each record, generate 3 additional paraphrase styles."""
    augmented = []
    missed = 0

    for rec in records:
        convs = rec["conversations"]
        if len(convs) < 2:
            continue

        original_text = convs[1]["content"]
        slots = extract_slots(original_text)

        # Structured fields by image path first (robust to any edit of the report
        # text, e.g. the histopathology removal); the text match is a fallback.
        us_report = lesion_lookup.get(lesion_key(rec["image_path"]) or "", {})
        if not us_report:
            us_report = response_to_report.get(_strip_tags(original_text), {})
        if not us_report and rec["source"] == "bus_cot":
            missed += 1

        # always keep the original
        augmented.append(rec)

        # add 3 paraphrase styles
        for style_fn, style_name in zip(STYLE_FUNCS, STYLE_NAMES):
            new_text = style_fn(slots, us_report)
            new_rec = {
                "id": f"{rec['id']}_{style_name}",
                "source": rec["source"],
                "image_path": rec["image_path"],
                "conversations": [
                    convs[0],
                    {"role": "assistant", "content": new_text},
                ],
                "metadata": {**rec.get("metadata", {}), "augmented": style_name},
            }
            augmented.append(new_rec)

    logger.info(
        "Augmented %d → %d records (%d BUS-CoT records without structured lookup)",
        len(records), len(augmented), missed,
    )
    return augmented


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified_dir", default="data/unified")
    parser.add_argument(
        "--lesion_json",
        default="data/raw/bus_cot/BUSCoT/DatasetFiles/lesion_dataset.json",
    )
    parser.add_argument("--out_dir", default="data/augmented")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build lookup: reasoning_response text → structured us_report
    logger.info("Building lesion lookup from %s", args.lesion_json)
    lesion_lookup = build_lesion_lookup(Path(args.lesion_json))
    response_to_report = build_filename_lookup(Path(args.lesion_json))
    logger.info("Lookup entries: %d by path, %d by text",
                len(lesion_lookup), len(response_to_report))

    for split in ["train", "val", "test"]:
        src = Path(args.unified_dir) / f"{split}.jsonl"
        if not src.exists():
            logger.warning("Missing %s, skipping", src)
            continue

        with open(src) as f:
            records = [json.loads(line) for line in f if line.strip()]

        if split == "train":
            records = augment_records(records, lesion_lookup, response_to_report)
            # Shuffle so augmented variants are mixed in
            random.shuffle(records)

        out_path = out_dir / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        logger.info("%s: %d records → %s", split, len(records), out_path)

    logger.info("Done. Use data/augmented/ in your training config.")


if __name__ == "__main__":
    main()
