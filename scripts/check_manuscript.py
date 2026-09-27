"""Integrity check for the revised manuscript, supplement and response letter.

1. Every decimal number in the prose must appear in the computed artifacts
   (outputs/stats/article_numbers.json and the other stats JSONs) at the
   precision it is quoted, or in a short list of design constants. Tables are
   generated from the same JSON (scripts/article_tables.py) and are skipped.
2. No `\\PENDING{...}` marker may remain (human input still missing).
3. No em dash (the authors' style rule): neither the Unicode character nor LaTeX `---`.

Unmatched numbers are printed with their line so each can be traced or fixed; the
script exits non-zero if anything fails.

Usage:
    python scripts/check_manuscript.py paper/main_body.tex \\
        paper/supplementary.tex \\
        paper/response_to_reviewers.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

ARTIFACTS = ["outputs/stats/article_numbers.json",
             "outputs/stats/statistics.json",
             "outputs/stats/decoder_scale.json",
             "outputs/stats/leakage_audit.json",
             "outputs/stats/leakage_submitted_split.json",
             "outputs/stats/leakage_intermediate_split.json",
             "outputs/stats/extractor_validation_templates.json",
             "outputs/stats/modality_gap.json",
             "outputs/stats/config_table.json",
             "outputs/extractor_validation/round1/extractor_validation.json",
             "outputs/extractor_validation/round2/extractor_validation.json"]
#: Design constants and settings quoted in the methods (not results).
CONSTANTS = {"0.05", "0.1", "0.02", "0.07", "1.0", "0.50", "0.90", "0.01", "0.95", "2.7",
             "0.5", "1.5", "5.1", "0.0", "1.000"}
_NUM = re.compile(r"(?<![\w\\{])(\d+(?:\{,\}\d{3})*(?:\.\d+)?)")


def _floats(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, dict):
        return [x for v in obj.values() for x in _floats(v)]
    if isinstance(obj, list):
        return [x for v in obj for x in _floats(v)]
    if isinstance(obj, str):
        return [float(m) for m in re.findall(r"-?\d+\.\d+", obj)]
    return []


def known_strings() -> set[str]:
    vals: list[float] = []
    for path in ARTIFACTS:
        if Path(path).exists():
            vals += _floats(json.loads(Path(path).read_text()))
    out = set(CONSTANTS)
    for v in vals:
        a = abs(v)
        for x in (a, a * 100):
            out |= {f"{x:.0f}", f"{x:.1f}", f"{x:.2f}", f"{x:.3f}", f"{x:.4f}"}
        if a > 0:  # slope-derived bounds quoted over the 0.5B-72B range (log10 144)
            out.add(f"{a * math.log10(144):.4f}")
    return out


def known_pairs() -> set[tuple[str, str]]:
    """(mean, sd) pairs as stored together in the numbers file, at 3 decimals."""
    pairs: set[tuple[str, str]] = set()

    def walk(obj) -> None:
        if isinstance(obj, dict):
            if isinstance(obj.get("mean"), float) and isinstance(obj.get("sd"), float):
                pairs.add((f"{obj['mean']:.3f}", f"{obj['sd']:.3f}"))
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)
    walk(json.loads(Path(ARTIFACTS[0]).read_text()))
    return pairs


_PAIR = re.compile(r"(\d\.\d{3})\s*\\pm\s*(\d\.\d{3})")


def check(path: Path, known: set[str], pairs: set[tuple[str, str]]) -> list[str]:
    problems = []
    text = path.read_text()
    for i, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("%") or stripped.startswith("\\input"):
            continue
        if "\\PENDING" in line and not stripped.startswith("\\newcommand"):
            problems.append(f"{path.name}:{i}: PENDING marker: {stripped[:100]}")
        is_rule = path.suffix == ".md" and stripped == "---"  # Markdown horizontal rule
        if "—" in line or (re.search(r"(?<!-)---(?!-)", line) and not is_rule):
            problems.append(f"{path.name}:{i}: em dash")
        # section numbers ("Section 4.10", "Sections 3.4") are not results
        line = re.sub(r"Sections? \d+(\.\d+)*", "", line)
        for mean, sd in _PAIR.findall(line):
            if (mean, sd) not in pairs:
                problems.append(f"{path.name}:{i}: {mean}+/-{sd} is not a stored mean/SD pair")
        for tok in _NUM.findall(line):
            norm = tok.replace("{,}", "")
            if "." not in norm:
                continue  # integers are counts/sizes; decimals are the risk
            if norm not in known:
                problems.append(f"{path.name}:{i}: {norm} not in artifacts: {stripped[:110]}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", type=Path)
    args = ap.parse_args()
    known, pairs = known_strings(), known_pairs()
    problems = [p for f in args.files for p in check(f, known, pairs)]
    print("\n".join(problems) if problems else "all numbers traced; no PENDING; no em dash")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
