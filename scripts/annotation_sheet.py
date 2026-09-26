"""Human validation of the regex slot extractor on *generated* reports.

Reviewer 1 item 15: the extractor was validated only on template-conformant
reference reports; perfect extraction there says nothing about model output.
This script (1) draws a stratified sample of generated reports across runs,
writes an annotation sheet with the extractor's output hidden, and (2) scores
the filled-in sheet(s): per-slot precision / recall of the extractor against
the human reading, and inter-annotator agreement (Cohen's kappa) when two
sheets are given.

The annotator reads each report and records what it *says* for every slot
(or "not stated"). No clinical judgement is involved: this checks whether the
regex reads text the way a person does.

Usage:
    # 1. sheet (100 reports, 20 per run, reproducible)
    python scripts/annotation_sheet.py make \\
        --runs g_cb3_dinov2_s1 g_cb9_dinov2_s1 g_dec_qwen7b_s1 g_enc_usfm_s1 g_vlm_qwen25vl_7b \\
        --per_run 20 --out outputs/annotation/sheet.csv
    #    -> sheet.csv (for the annotators) + sheet_key.json (run + extractor output, keep hidden)
    # 2. scoring
    python scripts/annotation_sheet.py score --key outputs/annotation/sheet_key.json \\
        --sheets outputs/annotation/sheet_A.csv outputs/annotation/sheet_B.csv \\
        --out outputs/annotation/extractor_validation.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.slot_labels import _MARGIN_SYNONYM_TO_ENUM  # noqa: E402
from src.evaluation.slots import _CALC_SYN, _ECHO_SYN, extract_slots  # noqa: E402

SLOTS = ["pathology", "birads", "orientation", "margins", "shape", "echogenicity", "calcification"]
NOT_STATED = "not stated"

#: Canonical values per slot. The annotator copies the words the report uses;
#: synonyms (any phrase of the four training styles) are folded onto these.
VOCAB = {
    "pathology": ["benign", "malignant"],
    "birads": ["2", "3", "4A", "4B", "4C", "5", "6"],
    "orientation": ["parallel", "not parallel"],
    "margins": ["regular (circumscribed / well-defined / smooth)",
                "partially regular (microlobulated / angular / partially circumscribed)",
                "irregular (indistinct / spiculated / poorly defined)"],
    "shape": ["oval", "round", "irregular"],
    "echogenicity": ["hypoechoic", "slightly hypoechoic", "isoechoic", "hyperechoic",
                     "heterogeneous", "complex cystic and solid", "anechoic"],
    "calcification": ["absent", "present"],
}


def _norm(value: str | None, slot: str) -> str:
    """Fold a human- or extractor-written value onto the canonical vocabulary."""
    v = (value or "").strip().lower()
    if not v or v in ("none", "n/a", "na", "-", NOT_STATED, "not stated", "missing"):
        return NOT_STATED
    if slot == "birads":
        return v.upper()
    if slot == "calcification":
        enum = _CALC_SYN.get(v, v)
        if enum == "nocalcification" or "absen" in v or v.startswith("no ") or v == "no":
            return "absent"
        return "present"
    if slot == "echogenicity":
        return _ECHO_SYN.get(v, v.replace(" echo", ""))
    if slot == "margins":
        return _MARGIN_SYNONYM_TO_ENUM.get(v, v.replace(" ", ""))
    return v


def extractor_reading(text: str) -> dict[str, str]:
    """The extractor's answer for each slot, in the sheet's canonical vocabulary
    (the extractor already returns enum keys for the descriptor slots)."""
    s = extract_slots(text)
    calc = s.get("calcification")
    return {
        "pathology": _norm(s.get("pathology"), "pathology"),
        "birads": _norm(s.get("birads"), "birads"),
        "orientation": _norm(s.get("orientation"), "orientation"),
        "margins": s.get("margins") or NOT_STATED,
        "shape": _norm(s.get("shape"), "shape"),
        "echogenicity": s.get("echogenicity") or NOT_STATED,
        "calcification": NOT_STATED if calc is None
        else ("absent" if calc == "nocalcification" else "present"),
    }


def make(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    rows, key = [], []
    for run in args.runs:
        path = Path("outputs/weekend") / run / "predictions.json"
        preds = json.load(open(path))
        for d in random.sample(preds, min(args.per_run, len(preds))):
            rows.append({"sample": None, "report": d["prediction"]})
            key.append({"run": run, "id": d["id"], "reference": d["reference"],
                        "extractor": extractor_reading(d["prediction"])})
    order = list(range(len(rows)))
    random.shuffle(order)  # runs are interleaved so the annotator cannot infer them
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "report"] + SLOTS)
        for n, i in enumerate(order, 1):
            w.writerow([f"S{n:03d}", rows[i]["report"]] + [""] * len(SLOTS))
    key_path = out.with_name(out.stem + "_key.json")
    key_path.write_text(json.dumps(
        {f"S{n:03d}": key[i] for n, i in enumerate(order, 1)}, indent=1))
    instructions = out.with_name(out.stem + "_INSTRUCTIONS.txt")
    instructions.write_text(
        "Extractor validation sheet\n"
        "==========================\n"
        "For each row, read the report and write what it STATES for each slot.\n"
        "Write 'not stated' if the report does not mention the slot. Do not judge\n"
        "whether the report is medically right; only record what the text says.\n\n"
        + "\n".join(f"{s}: {' / '.join(VOCAB[s])} / not stated" for s in SLOTS)
        + "\n\nCopy the words the report uses (e.g. 'low echogenicity', 'no calcific foci',\n"
        "'microlobulated'); any wording is fine, the scorer folds synonyms together.\n"
        "A report may be garbled or repeat itself: still record what it states, or\n"
        "'not stated' if a slot is absent or unreadable.\n"
        "Do not open the _key.json file until you have finished.\n")
    print(f"{len(rows)} reports -> {out}\nkey -> {key_path}\ninstructions -> {instructions}")


def _read_sheet(path: str) -> dict[str, dict[str, str]]:
    with open(path, newline="") as f:
        return {r["sample"]: {s: _norm(r.get(s), s) for s in SLOTS} for r in csv.DictReader(f)}


def _prf(pairs: list[tuple[str, str]]) -> dict:
    """Extractor vs human, treating 'not stated' as the negative class:
    precision = extracted values that the human agrees with; recall = human-read
    values the extractor recovered; exact = agreement including 'not stated'."""
    tp = sum(1 for e, h in pairs if e != NOT_STATED and e == h)
    fp = sum(1 for e, h in pairs if e != NOT_STATED and e != h)
    fn = sum(1 for e, h in pairs if h != NOT_STATED and e != h)
    exact = sum(1 for e, h in pairs if e == h)
    return {
        "n": len(pairs),
        "precision": tp / (tp + fp) if (tp + fp) else None,
        "recall": tp / (tp + fn) if (tp + fn) else None,
        "exact_agreement": exact / len(pairs) if pairs else None,
        "human_not_stated": sum(1 for _, h in pairs if h == NOT_STATED),
        "extractor_not_stated": sum(1 for e, _ in pairs if e == NOT_STATED),
    }


def _kappa(a: list[str], b: list[str]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def score(args: argparse.Namespace) -> None:
    key = json.load(open(args.key))
    sheets = [_read_sheet(p) for p in args.sheets]
    samples = [s for s in key if all(s in sh for sh in sheets)]
    out: dict = {"n_samples": len(samples), "sheets": args.sheets, "per_slot": {}, "per_run": {}}
    for slot in SLOTS:
        # the extractor against each annotator separately; kappa between the first two
        entry: dict = {"per_annotator": [
            _prf([(key[s]["extractor"][slot], sh[s][slot]) for s in samples]) for sh in sheets]}
        if len(sheets) > 1:
            a = [sheets[0][s][slot] for s in samples]
            b = [sheets[1][s][slot] for s in samples]
            entry["inter_annotator_kappa"] = _kappa(a, b)
            entry["inter_annotator_agreement"] = sum(x == y for x, y in zip(a, b)) / len(samples)
        out["per_slot"][slot] = entry
    for run in sorted({key[s]["run"] for s in samples}):
        ids = [s for s in samples if key[s]["run"] == run]
        out["per_run"][run] = {
            slot: [_prf([(key[s]["extractor"][slot], sh[s][slot]) for s in ids]) for sh in sheets]
            for slot in ("pathology", "birads")
        }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"n={len(samples)} reports, {len(sheets)} annotator(s)")
    for slot, e in out["per_slot"].items():
        pr = " ".join(f"P={p['precision']} R={p['recall']}" for p in e["per_annotator"])
        k = e.get("inter_annotator_kappa")
        print(f"  {slot:13s} {pr}" + (f" kappa={k:.3f}" if k is not None else ""))
    print(f"Saved -> {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--runs", nargs="+", required=True)
    m.add_argument("--per_run", type=int, default=20)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", default="outputs/annotation/sheet.csv")
    s = sub.add_parser("score")
    s.add_argument("--key", required=True)
    s.add_argument("--sheets", nargs="+", required=True)
    s.add_argument("--out", default="outputs/annotation/extractor_validation.json")
    args = ap.parse_args()
    make(args) if args.cmd == "make" else score(args)


if __name__ == "__main__":
    main()
