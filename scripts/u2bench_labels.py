"""Remove or restore U2-BENCH labels in the per-case output files.

U2-BENCH is licensed CC BY-NC-ND 4.0, which does not allow redistributing
modified copies of its records. The public release therefore ships our per-case
predictions for U2-BENCH frames without the dataset's labels, and without the
filtered record file `outputs/analysis7/u2bench_external.jsonl` (which
scripts/run_batch_analyses.sh rebuilds from the download). After downloading
U2-BENCH (scripts/download/_u2b_fetch.py), `restore` puts the labels back, so
every number and figure can be regenerated.

Labels are derived exactly as when the files were written: malignancy with
evaluate_u2bench.label_to_malignant (0 = benign, 1 = malignant), BI-RADS as the
record's class label.

Usage:
    python scripts/u2bench_labels.py strip   [--outputs outputs]
    python scripts/u2bench_labels.py restore [--outputs outputs] \\
        [--u2bench data/raw/u2bench/breast_eval/breast.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RESTORE_HINT = ("U2-BENCH labels are not redistributed (CC BY-NC-ND 4.0): download U2-BENCH with "
                "scripts/download/_u2b_fetch.py, then run scripts/u2bench_labels.py restore")


def _files(outputs: Path) -> dict[str, list[Path]]:
    a7 = outputs / "analysis7"
    probs = sorted(a7.glob("probs/*/u2bench.json")) + [outputs / "stats" / "probs_u2bench.json"]
    return {"probs": probs,
            "malignancy": sorted(a7.glob("u2b_malig/*/predictions.json")),
            "birads": sorted(a7.glob("u2b_birads/*/predictions_birads.json"))}


def _records(path: Path, data) -> list[dict]:
    return data["per_sample"] if path.name.endswith("u2bench.json") else data


def has_labels(outputs: Path = Path("outputs")) -> bool:
    probe = _files(outputs)["probs"][0]
    return "y" in _records(probe, json.load(open(probe)))[0]


def require_labels(outputs: Path = Path("outputs")) -> None:
    if not has_labels(outputs):
        raise SystemExit(RESTORE_HINT)


def _labels(u2bench: Path) -> tuple[dict[str, int], dict[str, str]]:
    from evaluate_u2bench import label_to_malignant  # imports torch; not needed to check labels
    malignant, birads = {}, {}
    for line in open(u2bench):
        r = json.loads(line)
        y = label_to_malignant(r["class_label"], r.get("options", ""), r["classification_task"])
        if y is not None:
            malignant[r["id"]] = y
        if r["classification_task"] == "BIRADS":
            birads[r["id"]] = r["class_label"]
    return malignant, birads


def _rewrite(path: Path, fn) -> None:
    data = json.load(open(path))
    for rec in _records(path, data):
        fn(rec)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def strip(outputs: Path) -> None:
    drop = {"probs": ("y",), "malignancy": ("gold", "gold_label"), "birads": ("gold",)}
    for kind, paths in _files(outputs).items():
        for p in paths:
            _rewrite(p, lambda r, keys=drop[kind]: [r.pop(k, None) for k in keys])
    (outputs / "analysis7" / "u2bench_external.jsonl").unlink(missing_ok=True)
    print(f"U2-BENCH labels removed under {outputs}")


def restore(outputs: Path, u2bench: Path) -> None:
    if not u2bench.exists():
        raise SystemExit(f"{u2bench} not found. {RESTORE_HINT}")
    from evaluate_u2bench import PATHOLOGY_CLASSES
    malignant, birads = _labels(u2bench)
    files = _files(outputs)
    for p in files["probs"]:
        _rewrite(p, lambda r: r.__setitem__("y", malignant[r["id"]]))
    for p in files["malignancy"]:
        _rewrite(p, lambda r: r.update(gold=malignant[r["id"]],
                                       gold_label=PATHOLOGY_CLASSES[malignant[r["id"]]]))
    for p in files["birads"]:
        _rewrite(p, lambda r: r.__setitem__("gold", birads[r["id"]]))
    print(f"U2-BENCH labels restored under {outputs} from {u2bench}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["strip", "restore"])
    ap.add_argument("--outputs", type=Path, default=Path("outputs"))
    ap.add_argument("--u2bench", type=Path,
                    default=Path("data/raw/u2bench/breast_eval/breast.jsonl"))
    args = ap.parse_args()
    if args.action == "strip":
        strip(args.outputs)
    else:
        restore(args.outputs, args.u2bench)


if __name__ == "__main__":
    main()
