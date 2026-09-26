"""Does a concept intervention change the whole report, or only the label slot?

Reviewer item 12: the reported intervention agreement (92.2% pathology / 85.5% risk)
only checks that the *forced* slot appears in the output. With an explicit
`<answer>` slot and an 8-token soft prompt derived from concept logits, that is close
to the floor for any working conditional decoder. The stronger question is whether the
descriptors CO-VARY: when pathology is flipped benign -> malignant, do
"circumscribed / oval" become "indistinct / irregular", or does only the label move?
If descriptors stay fixed, an intervened report is internally inconsistent, which
breaks the clinical override use case.

Design: for each probe image, generate a baseline report (no override) and one report
per forced pathology class. Compare descriptor slots between baseline and intervention,
separately for
  FLIP    - the forced class differs from the baseline report's own label
  CONTROL - the forced class equals it (descriptors should barely change)
A faithful, internally consistent model shows a high FLIP change-rate and a low
CONTROL change-rate.

Usage:
    python scripts/intervention_covariation.py \
        --checkpoint checkpoints/weekend/dinov2_cb/final.ckpt \
        --test_jsonl data/augmented_v2/test.jsonl --n 80 \
        --output outputs/stats/intervention_covariation.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.datasets.bus_cot_jepa import BUSCoTJEPADataset  # noqa: E402
from src.data.slot_labels import PATHOLOGY_CLASSES  # noqa: E402
from src.evaluation.slots import DESCRIPTOR_SLOTS, extract_slots  # noqa: E402
from src.model.vl_jepa import HistoVLJEPA  # noqa: E402

INV_PATH = {v: k for k, v in PATHOLOGY_CLASSES.items()}  # 0->benign, 1->malignant


def descriptors(text: str) -> dict[str, str | None]:
    s = extract_slots(text)
    return {k: (s.get(k) or None) for k in DESCRIPTOR_SLOTS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--test_jsonl", required=True)
    ap.add_argument("--n", type=int, default=80)
    # Match scripts/evaluate_jepa.py so co-variation is measured under the SAME
    # generation process as every other slot metric in the paper. With greedy
    # decoding at 128 tokens the 9-head model emits a prose surface form that the
    # style-A slot regex cannot parse, which silently produced null descriptors
    # (reported as an instrument limitation, not a finding, in an earlier run).
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = HistoVLJEPA.load_from_checkpoint(args.checkpoint, stage="finetune")
    model.eval().to(args.device)
    assert model.concept_bottleneck_enabled, "checkpoint is not a concept-bottleneck model"

    ds = BUSCoTJEPADataset(jsonl_path=args.test_jsonl, root_dir=Path("."),
                           image_size=224, preprocess_mode="ultrasound")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # counters, split by whether the intervention actually flips the baseline label
    tally = {
        arm: {"n": 0, "label_changed": 0,
              "any_descriptor_changed": 0,
              "base_unparsed": 0, "gen_unparsed": 0, "comparable": 0,
              "any_changed_comparable": 0,
              "per_slot_changed": {s: 0 for s in DESCRIPTOR_SLOTS},
              "per_slot_n": {s: 0 for s in DESCRIPTOR_SLOTS}}
        for arm in ("flip", "control")
    }
    examples: list[dict] = []
    seen = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Intervening"):
            if seen >= args.n:
                break
            images = batch["image"].to(args.device)
            B = images.shape[0]

            base_txt = model.forward_generate(images, max_new_tokens=args.max_new_tokens,
                                              num_beams=args.num_beams)
            base_slots = [extract_slots(t) for t in base_txt]
            base_desc = [descriptors(t) for t in base_txt]

            for cls, target in INV_PATH.items():
                ov = {"pathology": torch.full((B,), cls, dtype=torch.long,
                                              device=args.device)}
                gen = model.forward_generate(images, overrides=ov,
                                             max_new_tokens=args.max_new_tokens,
                                             num_beams=args.num_beams)
                for k, text in enumerate(gen):
                    base_label = (base_slots[k].get("pathology") or "").lower() or None
                    if base_label is None:
                        continue
                    arm = "control" if base_label == target.lower() else "flip"
                    t = tally[arm]
                    t["n"] += 1

                    got = extract_slots(text)
                    got_label = (got.get("pathology") or "").lower() or None
                    t["label_changed"] += int(got_label == target.lower())

                    new_desc = descriptors(text)
                    # A pair only enters a slot's denominator if BOTH sides parsed.
                    # Track the drops: an intervention that rewrites the report into a
                    # surface form the extractor misses would otherwise be silently
                    # excluded, biasing the change-rate downward.
                    t["base_unparsed"] += int(all(v is None for v in base_desc[k].values()))
                    t["gen_unparsed"] += int(all(v is None for v in new_desc.values()))
                    changed_any = False
                    comparable = 0
                    for s in DESCRIPTOR_SLOTS:
                        b, g = base_desc[k][s], new_desc[s]
                        if b is None or g is None:
                            continue
                        t["per_slot_n"][s] += 1
                        comparable += 1
                        if b.strip().lower() != g.strip().lower():
                            t["per_slot_changed"][s] += 1
                            changed_any = True
                    t["any_descriptor_changed"] += int(changed_any)
                    # A pair with no comparable slot cannot register a change, so
                    # including it in the denominator deflates the rate by exactly the
                    # unparseable fraction. Arms differ a lot in that fraction, so the
                    # headline comparison is only fair conditioned on comparability.
                    if comparable:
                        t["comparable"] += 1
                        t["any_changed_comparable"] += int(changed_any)

                    if arm == "flip" and len(examples) < 8:
                        examples.append({
                            "forced_pathology": target,
                            "baseline_label": base_label,
                            "baseline_descriptors": base_desc[k],
                            "intervened_descriptors": new_desc,
                            "intervened_report": text[:220],
                        })
            seen += B

    def rates(arm: str) -> dict:
        t = tally[arm]
        n = max(t["n"], 1)
        return {
            "n": t["n"],
            "forced_label_adopted_rate": round(t["label_changed"] / n, 4),
            "any_descriptor_changed_rate": round(t["any_descriptor_changed"] / n, 4),
            "per_slot_change_rate": {
                s: (round(t["per_slot_changed"][s] / t["per_slot_n"][s], 4)
                    if t["per_slot_n"][s] else None)
                for s in DESCRIPTOR_SLOTS
            },
            "per_slot_n": dict(t["per_slot_n"]),
            "baseline_fully_unparsed": t["base_unparsed"],
            "intervened_fully_unparsed": t["gen_unparsed"],
            "intervened_unparsed_rate": round(t["gen_unparsed"] / n, 4),
            "n_comparable": t["comparable"],
            "any_descriptor_changed_rate_comparable": (
                round(t["any_changed_comparable"] / t["comparable"], 4)
                if t["comparable"] else None),
        }

    res = {"n_images": seen, "checkpoint": args.checkpoint,
           "flip": rates("flip"), "control": rates("control"), "examples": examples}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(res, indent=2, ensure_ascii=False))

    print("\n=== Intervention co-variation (does the whole report move?) ===")
    for arm in ("flip", "control"):
        r = res[arm]
        print(f"  {arm.upper():8s} n={r['n']:4d} forced-label adopted "
              f"{r['forced_label_adopted_rate']:.3f} | any descriptor changed "
              f"{r['any_descriptor_changed_rate']:.3f} | comparable-only "
              f"{r['any_descriptor_changed_rate_comparable']} (n={r['n_comparable']}) "
              f"| intervened text unparseable "
              f"{r['intervened_unparsed_rate']:.3f}")
        for s, v in r["per_slot_change_rate"].items():
            print(f"      {s:13s} change-rate {v}  (n={r['per_slot_n'][s]})")
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
