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
    #    -> sheet.csv + sheet_A.html / sheet_B.html (for the annotators) and
    #       sheet_key.json (run + extractor output, keep hidden)
    #    Each annotator opens their .html file in a browser, clicks through the reports
    #    (answers are kept in the browser) and sends back the sheet_<A|B>.csv it downloads.
    # 2. scoring
    python scripts/annotation_sheet.py score --key outputs/annotation/sheet_key.json \\
        --sheets outputs/annotation/sheet_A.csv outputs/annotation/sheet_B.csv \\
        --out outputs/annotation/extractor_validation.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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


def write_html(sheet_rows: list[dict], annotator: str, path: Path) -> None:
    """Clickable sheet (scripts/annotation_tool.html) with the reports embedded and the
    extractor output left out; it downloads sheet_<annotator>.csv in the sheet.csv format."""
    sheet_id = hashlib.sha256(json.dumps(sheet_rows).encode()).hexdigest()[:12]
    data = {"id": sheet_id, "annotator": annotator, "slots": SLOTS, "vocab": VOCAB,
            "rows": sheet_rows}
    # "<" escaped so that no report text can close the <script> element
    blob = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    template = (Path(__file__).parent / "annotation_tool.html").read_text()
    assert template.count("/*__SHEET__*/null") == 1
    path.write_text(template.replace("/*__SHEET__*/null", blob))


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
    sheet_rows = [{"sample": f"S{n:03d}", "report": rows[i]["report"]}
                  for n, i in enumerate(order, 1)]
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "report"] + SLOTS)
        for r in sheet_rows:
            w.writerow([r["sample"], r["report"]] + [""] * len(SLOTS))
    for a in args.annotators:
        write_html(sheet_rows, a, out.with_name(f"{out.stem}_{a}.html"))
    key_path = out.with_name(out.stem + "_key.json")
    key_path.write_text(json.dumps(
        {f"S{n:03d}": key[i] for n, i in enumerate(order, 1)}, indent=1))
    instructions = out.with_name(out.stem + "_INSTRUCTIONS.txt")
    instructions.write_text(
        "Extractor validation sheet\n"
        "==========================\n"
        "Easiest: open your sheet_<A or B>.html file in a web browser, click through the\n"
        "reports and press 'Download CSV' at the end. The rules below apply either way.\n\n"
        "For each row, read the report and write what it STATES for each slot.\n"
        "Write 'not stated' if the report does not mention the slot. Do not judge\n"
        "whether the report is medically right; only record what the text says.\n\n"
        + "\n".join(f"{s}: {' / '.join(VOCAB[s])} / not stated" for s in SLOTS)
        + "\n\nCopy the words the report uses (e.g. 'low echogenicity', 'no calcific foci',\n"
        "'microlobulated'); any wording is fine, the scorer folds synonyms together.\n"
        "A report may be garbled or repeat itself: still record what it states, or\n"
        "'not stated' if a slot is absent or unreadable.\n"
        "Do not open the _key.json file until you have finished.\n")
    print(f"{len(rows)} reports -> {out} and "
          + ", ".join(f"{out.stem}_{a}.html" for a in args.annotators)
          + f"\nkey -> {key_path}\ninstructions -> {instructions}")


def _read_sheet(path: str, reports: dict[str, str]) -> dict[str, dict[str, str]]:
    """Normalized answers per sample of a returned sheet, checked against the blank sheet it
    was made from (`reports`: sample -> report text). Sample ids repeat across rounds
    (S001, ...), so the sheet must hold exactly this round's samples with the same report
    text. A blank cell is an unfinished answer, not "not stated" (the clickable sheet
    writes "not stated" explicitly), so a sheet with blanks is refused."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["sample"] for r in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(reports):
        sys.exit(f"{path}: samples do not match the blank sheet ({len(set(ids))} unique of "
                 f"{len(ids)} rows, {len(set(ids) & set(reports))} of {len(reports)} expected)")
    wrong = [r["sample"] for r in rows if r["report"] != reports[r["sample"]]]
    if wrong:
        sys.exit(f"{path}: report text differs from the blank sheet for {len(wrong)} sample(s), "
                 f"e.g. {', '.join(wrong[:5])}: a sheet of another round?")
    blank = [r["sample"] for r in rows if any(not (r.get(s) or "").strip() for s in SLOTS)]
    if blank:
        sys.exit(f"{path}: {len(blank)} report(s) not finished, e.g. {', '.join(blank[:5])}")
    return {r["sample"]: {s: _norm(r.get(s), s) for s in SLOTS} for r in rows}


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
    blank = args.sheet or str(Path(args.key).with_name(Path(args.key).stem[:-len("_key")] + ".csv"))
    with open(blank, newline="") as f:
        reports = {r["sample"]: r["report"] for r in csv.DictReader(f)}
    if set(reports) != set(key):
        sys.exit(f"{blank} and {args.key} list different samples")
    sheets = [_read_sheet(p, reports) for p in args.sheets]
    samples = list(key)
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
    m.add_argument("--annotators", nargs="+", default=["A", "B"],
                   help="one clickable sheet_<name>.html per annotator")
    s = sub.add_parser("score")
    s.add_argument("--key", required=True)
    s.add_argument("--sheets", nargs="+", required=True)
    s.add_argument("--sheet", default=None,
                   help="blank sheet the returned ones were made from (default: the .csv "
                        "next to --key)")
    s.add_argument("--out", default="outputs/annotation/extractor_validation.json")
    args = ap.parse_args()
    make(args) if args.cmd == "make" else score(args)


if __name__ == "__main__":
    main()
