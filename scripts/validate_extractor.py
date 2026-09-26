"""Validate the slot extractor against the dataset's own structured fields, in
every report style the models are trained to produce.

Every clinical number in the paper is a label recovered from generated text, so
the extractor is the measurement instrument and needs its own validation.
BUS-CoT ships the structured fields the reports were templated from
(`DatasetFiles/lesion_dataset.json`), so for every test record we (1) run
`extract_slots` on the reference report (style A) and (2) re-render the same
record in the three paraphrase styles (B clinical note, C prose, D brief) with
the training-time renderer and run the extractor on each. Comparing against the
structured fields measures the instrument on template-conformant text in all
four styles, with no annotation and no circularity (the fields are the source,
the regex is the instrument under test). Extraction on unconstrained model text
is the separate, human-annotated check of scripts/annotation_sheet.py.

Validated slots (unambiguous structured ground truth):
  pathology      <- pathology_histology.pathology       (Benign/Malignant)
  birads         <- us_report.BIRADS                    (2, 3, 4A, 4B, 4C, 5)
  echogenicity   <- us_report.EchoCharacteristics       (7 enums)
  calcification  <- us_report.LesionCalcificationFeatures (6 enums)
  shape, orientation, margins <- the source reasoning text. Margins deliberately
      NOT against us_report.LesionEdge: in BUS-CoT the margin word of the text and
      the LesionEdge enum are independent annotations that agree in only ~82% of
      records (e.g. 300 records say "circumscribed" under LesionEdge=PartiallyRegular),
      so the text word, folded onto the enum, is the extractor's ground truth.

Usage:
    python scripts/validate_extractor.py --split-file data/unified_v4g/test_buscot_only.jsonl \\
        --output outputs/stats/extractor_validation_v4g.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "preprocess"))
import build_augmented as renderer  # noqa: E402

from src.data.slot_labels import _MARGIN_SYNONYM_TO_ENUM, lesion_key  # noqa: E402
from src.evaluation.slots import extract_slots  # noqa: E402

LESION_JSON = Path("data/raw/bus_cot/BUSCoT/DatasetFiles/lesion_dataset.json")
SLOTS = ("pathology", "birads", "margins", "echogenicity", "calcification", "shape", "orientation")
_ORI_RE = re.compile(r"This lesion is (not parallel|parallel)", re.IGNORECASE)
_SHA_RE = re.compile(r"margins and (\w+) shape", re.IGNORECASE)
_MAR_RE = re.compile(r"has (\w+) margins", re.IGNORECASE)


def _norm(v) -> str | None:
    s = str(v).strip().lower().replace(" ", "") if v is not None else ""
    return s or None


def truth_for(entry: dict) -> dict[str, str | None]:
    """Ground truth per slot from the structured record (enum keys, lower-case)."""
    ph = entry.get("pathology_histology") or {}
    us = entry.get("us_report") or {}
    rr = us.get("reasoning_response") or ""
    ori = _ORI_RE.search(rr)
    sha = _SHA_RE.search(rr)
    mar = _MAR_RE.search(rr)
    return {
        "pathology": _norm(ph.get("pathology")),
        "birads": (str(us.get("BIRADS")).strip().upper() or None) if us.get("BIRADS") else None,
        "margins": _MARGIN_SYNONYM_TO_ENUM.get(mar.group(1).lower()) if mar else None,
        "echogenicity": _norm(us.get("EchoCharacteristics")),
        "calcification": _norm(us.get("LesionCalcificationFeatures")),
        "shape": _norm(sha.group(1)) if sha else None,
        "orientation": _norm(ori.group(1)) if ori else None,
    }


def renderings(ref_text: str, us_report: dict) -> dict[str, str]:
    slots = renderer.extract_slots(ref_text)
    return {"A": ref_text, "B": renderer.style_B(slots, us_report),
            "C": renderer.style_C(slots, us_report), "D": renderer.style_D(slots, us_report)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split-file", default="data/unified_v4g/test_buscot_only.jsonl")
    ap.add_argument("--output", default="outputs/stats/extractor_validation_v4g.json")
    args = ap.parse_args()

    lesions = json.load(open(LESION_JSON))
    records = [json.loads(line) for line in open(args.split_file) if line.strip()]

    stat: dict[str, dict[str, Counter]] = {st: {s: Counter() for s in SLOTS} for st in "ABCD"}
    mismatches: dict[str, list] = {s: [] for s in SLOTS}
    n_matched = n_unmatched = 0
    for rec in records:
        entry = lesions.get(lesion_key(rec.get("image_path", "")) or "")
        if entry is None:
            n_unmatched += 1
            continue
        n_matched += 1
        truth = truth_for(entry)
        texts = renderings(rec["conversations"][-1]["content"], entry.get("us_report") or {})
        for style, text in texts.items():
            got = extract_slots(text)
            for s in SLOTS:
                t = truth.get(s)
                g = got.get(s)
                g = g.upper() if (s == "birads" and g) else (_norm(g) if g else None)
                if t is None:
                    stat[style][s]["no_ground_truth"] += 1
                    continue
                stat[style][s]["with_ground_truth"] += 1
                if g is None:
                    stat[style][s]["not_extracted"] += 1
                elif g == t:
                    stat[style][s]["correct"] += 1
                else:
                    stat[style][s]["wrong"] += 1
                    if len(mismatches[s]) < 10:
                        mismatches[s].append({"id": rec["id"], "style": style, "truth": t,
                                              "extracted": g, "text": text[:160]})

    out: dict = {"split_file": args.split_file, "n_records": n_matched,
                 "n_unmatched": n_unmatched, "styles": {}, "example_mismatches": mismatches}
    for style in "ABCD":
        out["styles"][style] = {}
        for s in SLOTS:
            c = stat[style][s]
            wgt = c["with_ground_truth"]
            extracted = c["correct"] + c["wrong"]
            out["styles"][style][s] = {
                "with_ground_truth": wgt,
                "extraction_rate": round(extracted / wgt, 4) if wgt else None,
                "accuracy": round(c["correct"] / extracted, 4) if extracted else None,
                "correct": c["correct"], "wrong": c["wrong"],
                "not_extracted": c["not_extracted"],
            }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2))

    print(f"=== Extractor validation vs structured fields, {n_matched} records x 4 styles ===")
    print(f"{'slot':14s}" + "".join(f"{st:>22s}" for st in "ABCD"))
    for s in SLOTS:
        row = f"{s:14s}"
        for st in "ABCD":
            d = out["styles"][st][s]
            row += (f"   rate {d['extraction_rate']:.3f} acc {d['accuracy']:.3f}"
                    if d["extraction_rate"] is not None else f"{'n/a':>22s}")
        print(row)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
