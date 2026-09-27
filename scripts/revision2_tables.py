"""LaTeX tables of the revised manuscript and supplement (revision 2).

Every cell is read from outputs/stats/revision2_numbers.json (scripts/revision2_numbers.py)
or from the batch-7 per-run artifacts, so no number is transcribed by hand.
Encoder properties (architecture, pretraining data, objective) are the one piece of
curated metadata; their sources are the cited model papers and model cards.

Usage:
    python scripts/revision2_tables.py --outdir paper/frontiers/revision2/tables
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

NUM = Path("outputs/stats/revision2_numbers.json")
CHECK = " $\\checkmark$"


def f3(x: float) -> str:
    return f"{x:.3f}"


def msd(b: dict) -> str:
    return f"{b['mean']:.3f}" if b.get("sd") is None else f"{b['mean']:.3f}$\\pm${b['sd']:.3f}"


def sg(x: float, d: int = 3) -> str:
    """Signed number with a typeset minus sign."""
    return f"${x:+.{d}f}$"


def rng(b: dict) -> str:
    return f"{b['min']:.3f}--{b['max']:.3f}"


def table(caption: str, label: str, colspec: str, header: str, body: list[str],
          note: str | None = None, star: bool = False, size: str = "\\footnotesize",
          fit: bool = False) -> str:
    env = "table*" if star else "table"
    # fit: scale to the text width like the manuscript's other tables (main-text tables),
    # so wide ones do not overflow and narrow ones match the body text size.
    tab = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", header + " \\\\", "\\midrule",
           *body, "\\bottomrule", "\\end{tabular}"]
    if fit:
        tab = ["\\resizebox{\\linewidth}{!}{%", *tab[:-1], "\\end{tabular}%", "}"]
    out = [f"\\begin{{{env}}}[h!]", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\centering", size, *tab]
    if note:
        out.append(f"\\par\\vspace{{2pt}}\\parbox{{\\linewidth}}{{\\scriptsize {note}}}")
    out.append(f"\\end{{{env}}}")
    return "\n".join(out) + "\n"


# ── Table 1: headline comparison ──────────────────────────────────────────
def tab_main(n: dict) -> str:
    rows = n["main_table"]["rows"]
    auc = {m: n["analyses"]["models"][m]["auroc"]["test"] for m in ("cb3", "cb9")}
    probe = n["analyses"]["linear_probe"]
    spec = [("CB-3 (DINOv2-L, three assessment concepts)",
             "DINOv2-L + 3-head concept bottleneck", auc["cb3"]),
            ("CB-9 (DINOv2-L, assessment + six finding concepts)",
             "DINOv2-L + 9-head (finding-level) bottleneck", auc["cb9"]),
            ("Opaque bottleneck (DINOv2-L, no concept routing)", "DINOv2 ViT-L/14 (opaque)", None)]
    body = []
    for label, key, a in spec:
        r = rows[key]
        body.append(f"{label} & {len(r['runs'])} & {msd(r['path'])} & {msd(r['risk'])} & "
                    f"{msd(r['exact'])} & {msd(r['ab'])} & {msd(a) if a else 'n/a'} \\\\")
    body.append("\\midrule")
    body.append("Linear probe on frozen DINOv2-L features & n/a & "
                f"{f3(probe['pathology']['f1'])} & {f3(probe['risk']['f1'])} & "
                "n/a & n/a & n/a \\\\")
    body.append("\\midrule")
    body.append("\\multicolumn{7}{l}{\\emph{End-to-end VLM reference baselines "
                "(LoRA; compute and tuning parity not established)}} \\\\")
    for key in ("Qwen2.5-VL-7B (end-to-end LoRA)", "InternVL3-8B (end-to-end LoRA)",
                "Qwen2-VL-7B (end-to-end LoRA)"):
        r = rows[key]
        body.append(f"{key.split(' (')[0]} & {len(r['runs'])} & {msd(r['path'])} & "
                    f"{msd(r['risk'])} & {msd(r['exact'])} & {msd(r['ab'])} & n/a \\\\")
    return table(
        "Headline results on the official BUS-CoT test split ($n=873$ lesion crops, 486 patient "
        "groups, no patient or near-duplicate frame shared with training or validation). Mean "
        "$\\pm$ SD over seeds. Malignancy and risk-group F1 are read from the generated report "
        "by the validated extractor (coverage $\\geq$"
        f"{n['comparisons']['min_extraction_coverage']:.3f}"
        " in every run). ``Exact'' is exact accuracy over the seven BI-RADS-like categories and "
        "``4A vs 4B'' exact accuracy on references in category 4A or 4B (unextractable counts as "
        "wrong). AUROC is from the concept head's malignancy probability. The linear probe is "
        "deterministic, so it has no seed spread.",
        "tab:main", "lcccccc",
        "Configuration & Seeds & Malignancy F1 & Risk-group F1 & Exact category & 4A vs 4B & "
        "AUROC", body, star=True, fit=True)


# ── Table 2: decoder sweep ────────────────────────────────────────────────
SIZES = [("0.5B", "0_5b"), ("1.5B", "1_5b"), ("3B", "3b"), ("7B", "7b"), ("14B", "14b"),
         ("32B", "32b"), ("72B", "72b")]


def sweep_batch_sizes() -> str:
    """Stage-2 batch size per decoder size, from the job list that trained the sweep."""
    import sys

    from omegaconf import OmegaConf
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_weekend6 as b6
    base = OmegaConf.load("configs/train/finetune_jepa_v5.yaml").train.batch_size
    groups: dict[int, list[str]] = {}
    for tag, _, extra, _, _ in b6.QWEN:
        bs = next((int(e.split("=")[1]) for e in extra if e.startswith("train.batch_size=")),
                  base)
        groups.setdefault(bs, []).append(tag.replace("_", ".").upper())

    def join(x: list[str]) -> str:
        return x[0] if len(x) == 1 else ", ".join(x[:-1]) + " and " + x[-1]
    parts = [f"{bs} for {join(tags)}" for bs, tags in groups.items()]
    return ", ".join(parts[:-1]) + ", and " + parts[-1] if len(parts) > 1 else parts[0]


def tab_decoders(n: dict) -> str:
    rows = n["main_table"]["rows"]
    pairs = n["decoder"]["pairs"]

    def delta(tag: str, ep: str) -> str:
        if tag == "0_5b":
            return "reference"
        p = pairs[f"{tag}_vs_0_5b_s1"][ep]
        return (f"{sg(p['delta_f1'])} [{sg(p['ci90'][0])}, {sg(p['ci90'][1])}]"
                + (" (equiv.)" if p["equivalent_at_margin"] else ""))
    body = []
    for label, tag in SIZES:
        key = f"Qwen2.5-{label}" + (" (4-bit)" if tag == "72b" else "")
        r = rows[key]
        body.append(f"{label} & {len(r['runs'])} & {msd(r['path'])} & {msd(r['risk'])} & "
                    f"{msd(r['exact'])} & {delta(tag, 'pathology')} & {delta(tag, 'risk')} \\\\")
    boot = n["decoder"]["dose_response_patient_bootstrap"]
    ols = n["decoder"]["dose_response"]

    def slope(ep: str) -> str:
        b, o = boot[ep], ols[ep]
        lo, hi = b["ci95_patient_bootstrap"]
        return (f"{sg(b['slope_per_decade'], 4)} (patient-cluster bootstrap 95\\% CI "
                f"{sg(lo, 4)} to {sg(hi, 4)}; across runs {sg(o['ci95'][0], 4)} to "
                f"{sg(o['ci95'][1], 4)})")
    return table(
        "Single-family decoder sweep (Qwen2.5-Instruct, 0.5B--72B). Every decoder is conditioned "
        "only on the three assessment concepts of CB-3, starting Stage 2 from the same Stage-1 "
        "checkpoint (CB-3, seed 1), with the same LoRA settings ($r{=}16$, $\\alpha{=}32$, "
        "dropout 0.05). Decoder size is not the only difference: the Stage-2 batch size is "
        f"{sweep_batch_sizes()} (no gradient accumulation), the 72B decoder is loaded in 4 bits, "
        "and the three largest decoders have one seed. $\\Delta$ is the F1 difference to the "
        "0.5B decoder at seed 1 with its 90\\% patient-cluster bootstrap interval; ``equiv.'' "
        "would mark an interval inside the pre-specified $\\pm0.02$ margin (TOST; "
        f"established in {n['decoder']['tost_equivalent'].replace('/', ' of ')} size, seed and "
        "endpoint comparisons, Supplementary Table S7). "
        "Per tenfold increase in parameters, fitted over all 15 runs, malignancy F1 changes by "
        f"{slope('path_f1')} and risk-group F1 by {slope('risk_f1')}. The bootstrap interval "
        "resamples test patients with the trained models held fixed; the across-run interval "
        "treats the 15 runs as independent and so reflects training variability.",
        "tab:decoders", "lcccccc",
        "Decoder & Seeds & Malignancy F1 & Risk-group F1 & Exact category & "
        "$\\Delta$ malignancy [90\\% CI] & $\\Delta$ risk group [90\\% CI]", body, fit=True)


# ── Table 3: encoders with their properties ───────────────────────────────
ENC = [  # label in batch7_table, display, architecture, pretraining data, objective
    ("DINOv2 ViT-L/14 (opaque)", "DINOv2", "ViT-L/14", "LVD-142M natural images",
     "self-distillation + MIM"),
    ("DINOv3 ViT-L/16", "DINOv3", "ViT-L/16", "LVD-1689M natural images",
     "self-distillation + MIM"),
    ("RAD-DINO ViT-B/14 (chest X-ray)", "RAD-DINO", "ViT-B/14", "$\\sim$0.84M chest X-rays",
     "self-distillation + MIM"),
    ("DINOv2 ViT-B/14", "DINOv2-B", "ViT-B/14", "LVD-142M natural images",
     "self-distillation + MIM"),
    ("EVA-02 ViT-L/14 (MIM)", "EVA-02", "ViT-L/14", "ImageNet-22k", "MIM (CLIP-feature targets)"),
    ("USFM ViT-B/16 (ultrasound)", "USFM", "ViT-B/16", "$>$2M ultrasound images",
     "masked image modeling"),
    ("SigLIP ViT-L/16 (image-text)", "SigLIP", "ViT-L/16", "WebLI image--text pairs",
     "sigmoid contrastive"),
    ("ImageNet-21k ViT-L/16 (supervised)", "ImageNet-21k", "ViT-L/16", "ImageNet-21k",
     "supervised classification"),
    ("UNI2-h ViT-H/14 (histopathology)", "UNI2-h", "ViT-H/14",
     "$>$200M histopathology tiles", "self-distillation + MIM"),
]


def tab_encoders(n: dict) -> str:
    rows = n["main_table"]["rows"]
    body = []
    for key, name, arch, data, obj in ENC:
        r = rows[key]
        body.append(f"{name} & {arch} & {data} & {obj} & {msd(r['path'])} & {msd(r['risk'])} \\\\")
    return table(
        "Frozen image encoders (opaque bottleneck, BioMedLM-2.7B decoder, three seeds each), with "
        "the properties that differ between them. Every encoder receives 224-pixel input "
        "(position embeddings interpolated where the native resolution differs), so input "
        "resolution is controlled; architecture size, patch size (hence token count), pretraining "
        "data and objective are not. DINOv2-B is included as the size-matched (ViT-B) "
        "comparator for the ultrasound-pretrained USFM. UNI2-h shares the DINOv2 training "
        "objective but not its data domain.",
        "tab:encoders", "llllcc",
        "Encoder & Arch. & Pretraining data & Objective & Malignancy F1 & Risk-group F1",
        body, star=True, fit=True)


# ── Table 4: discrimination, calibration and operating point ─────────────
def tab_external(n: dict) -> str:
    a = n["analyses"]["models"]
    ext = n["data"]["external_in_train_val"]
    names = {"test": "Internal, BUS-CoT test ($n=873$)",
             "u2bench": "External, U2-BENCH ($n=294$)",
             "breast": "External, BrEaST ($n=252$)"}
    body = []
    for key, label in names.items():
        for m, mname in (("cb3", "CB-3"), ("cb9", "CB-9")):
            x = a[m]
            body.append(f"{label if m == 'cb3' else ''} & {mname} & {msd(x['auroc'][key])} & "
                        f"{rng(x['ece'][key])} & {msd(x['op90'][key]['sens'])} & "
                        f"{msd(x['op90'][key]['spec'])} & {msd(x['argmax'][key]['sens'])} & "
                        f"{msd(x['argmax'][key]['spec'])} \\\\")
        if key != "breast":
            body.append("\\addlinespace")
    return table(
        "Malignancy discrimination, calibration and operating point of the concept head, "
        "internally and on two external sets never used in training (mean $\\pm$ SD over three "
        "seeds; ECE as the seed range). U2-BENCH is restricted to its malignancy-labelled frames "
        f"that are not in BUS-CoT training or validation: {ext['u2bench']['in_train_val']} of its "
        f"{ext['u2bench']['n']} breast frames are BUSI images also contained in BUS-CoT and were "
        "removed. The 90\\%-sensitivity threshold is chosen on the internal validation split and "
        "applied unchanged to test and external data.",
        "tab:external", "llcccccc",
        "Evaluation set & Model & AUROC & ECE & \\multicolumn{2}{c}{Val-chosen threshold} & "
        "\\multicolumn{2}{c}{Argmax threshold} \\\\\n & & & & Sens. & Spec. & Sens. & Spec.",
        body, star=True, fit=True)


# ── Table 5: external report-level evaluation ─────────────────────────────
def tab_external_reports(n: dict) -> str:
    a = n["analyses"]["models"]
    u2b = []
    for m, mname in (("cb3", "CB-3"), ("cb9", "CB-9")):
        b = a[m]["u2b_birads"]
        for src, label in (("concept_head", "concept head"),
                           ("generated_report", "generated report")):
            c = b[src]
            u2b.append(f"{mname if src == 'concept_head' else ''} & {label} & "
                       f"{msd(c['exact'])} & {msd(c['ab'])} & {msd(c['risk_f1'])} \\\\")
    cols = ("shape", "margin", "echogenicity", "calcification")
    br = []
    for m, mname in (("cb3", "CB-3"), ("cb9", "CB-9")):
        b = a[m]["breast"]
        br.append(f"{mname} & accuracy & " + " & ".join(msd(b[c]) for c in cols)
                  + f" & {msd(b['birads_exact'])} & {msd(b['path_f1'])} \\\\")
        br.append(" & balanced accuracy & " + " & ".join(msd(b[f"{c}_balanced"]) for c in cols)
                  + f" & {msd(b['birads_balanced'])} & \\\\")
    base = n["analyses"]["breast_baselines"]
    br += ["\\midrule",
           "Image-blind & majority class & "
           + " & ".join(f3(base[c]["majority_accuracy"]) for c in (*cols, "birads_exact"))
           + f" & {f3(base['path_f1_always_malignant'])} \\\\",
           " & chance & "
           + " & ".join(f3(base[c]["chance_balanced_accuracy"]) for c in (*cols, "birads_exact"))
           + " & \\\\"]
    lines = [
        "\\begin{table*}[h!]",
        "\\caption{External evaluation of the generated report, not only of the classifier "
        "(mean $\\pm$ SD over three seeds). \\textbf{(A)} U2-BENCH BI-RADS-like subset ($n=109$, "
        "categories 2--5, no overlap with training): the category read from the concept head and "
        "from the generated report. \\textbf{(B)} BrEaST ($n=252$): agreement of the generated "
        "report's descriptors and category with the radiologist's BI-RADS lexicon annotation "
        "(missing counts as wrong), as accuracy and as balanced accuracy (mean recall over the "
        "annotated classes), and malignancy F1 of the report. Margin is scored as circumscribed "
        "versus not circumscribed and calcification as absent versus present. The image-blind "
        "rows give the accuracy of a report that always states the most frequent annotated class "
        "(for malignancy F1: always malignant) and the chance level of balanced accuracy.}",
        "\\label{tab:external_reports}", "\\centering", "\\footnotesize",
        "\\begin{tabular}{llccc}", "\\toprule",
        "\\multicolumn{5}{l}{\\textbf{(A)} U2-BENCH} \\\\",
        "Model & Source & Exact category & 4A vs 4B & Risk-group F1 \\\\", "\\midrule", *u2b,
        "\\bottomrule", "\\end{tabular}", "\\par\\vspace{6pt}",
        "\\begin{tabular}{llcccccc}", "\\toprule",
        "\\multicolumn{8}{l}{\\textbf{(B)} BrEaST, generated report} \\\\",
        "Model & Score & Shape & Margin & Echogenicity & Calcification & Exact category & "
        "Malignancy F1 \\\\", "\\midrule", *br, "\\bottomrule", "\\end{tabular}",
        "\\end{table*}"]
    return "\n".join(lines) + "\n"


# ── Supplement ────────────────────────────────────────────────────────────
def supp_runs() -> str:
    stats = json.load(open("outputs/stats/stats_batch7.json"))["per_run"]
    body = []
    # latest ok row per run (a regenerated run is appended to the CSV), in first-run order
    latest = {r["name"]: r for r in csv.DictReader(open("outputs/weekend7_results.csv"))
              if r["status"] == "ok"}
    for r in latest.values():
        s = stats.get(r["name"])
        ci = (lambda k: f"[{s[k]['ci95_cluster'][0]:.3f}, {s[k]['ci95_cluster'][1]:.3f}]") \
            if s else (lambda k: "n/a")
        cov = (f"{min(s['path_f1']['coverage'], s['risk_f1']['coverage']):.3f}" if s else "1.000")
        name = r["name"].removeprefix("h_").replace("_", "\\_")
        body.append(f"{name} & {float(r['path_f1']):.3f} {ci('path_f1')} & "
                    f"{float(r['risk_f1']):.3f} {ci('risk_f1')} & {cov} & "
                    f"{float(r['bleu4']):.3f} \\\\")
    head = ("\\toprule\nRun & Malignancy F1 [95\\% CI] & Risk-group F1 [95\\% CI] & Coverage & "
            "BLEU-4 \\\\\n\\midrule\n")
    return ("\\begin{longtable}{lcccc}\n\\caption{Every batch-7 run on the official BUS-CoT test "
            "split: F1 with 95\\% patient-cluster bootstrap interval, extraction coverage "
            "(lower of the two endpoints) and BLEU-4. No run falls below the pre-specified "
            "coverage thresholds (exclusion $<0.50$, flag $<0.90$).}"
            "\\label{tab:s_runs}\\\\\n" + head + "\\endfirsthead\n" + head + "\\endhead\n"
            + "\n".join(body) + "\n\\bottomrule\n\\end{longtable}\n")


def supp_comparisons(n: dict) -> str:
    c = n["comparisons"]["pairs"]
    holm = n["comparisons"]["holm_family"]["adjusted_p"]
    labels = [("DINOv2vsUNI2h", "DINOv2-L $-$ UNI2-h", True),
              ("DINOv2BvsUSFM", "DINOv2-B $-$ USFM", True),
              ("CB3vsOpaque", "CB-3 $-$ opaque", True), ("CB9vsOpaque", "CB-9 $-$ opaque", True),
              ("CB9vsCB3", "CB-9 $-$ CB-3", False), ("DINOv2vsUSFM", "DINOv2-L $-$ USFM", False),
              ("DINOv2vsIN21k", "DINOv2-L $-$ ImageNet-21k", False),
              ("CB3vs_qwen25vl_7b", "CB-3 $-$ Qwen2.5-VL-7B", False),
              ("CB9vs_qwen25vl_7b", "CB-9 $-$ Qwen2.5-VL-7B", False),
              ("CB3vs_internvl3_8b", "CB-3 $-$ InternVL3-8B", False),
              ("CB9vs_internvl3_8b", "CB-9 $-$ InternVL3-8B", False)]
    body = []
    for key, label, fam in labels:
        for s in (1, 2, 3):
            v = c[f"{key}_s{s}"]
            cells = []
            for ep in ("pathology", "risk"):
                e = v[ep]
                h = holm[f"s{s}"].get(f"{key}_{ep}") if fam else None
                p = f"{h:.3f}" if h is not None else f"{e['mcnemar_p_one_per_patient']:.3f}$^e$"
                cells.append(f"{e['delta']:+.3f} [{e['ci95_cluster'][0]:+.3f}, "
                             f"{e['ci95_cluster'][1]:+.3f}] & {p}")
            body.append(f"{label if s == 1 else ''} & {s} & {' & '.join(cells)} \\\\")
        body.append("\\addlinespace")
    return table(
        "Paired comparisons per seed: F1 difference with 95\\% patient-cluster bootstrap interval "
        "and McNemar $p$ on one record per patient. The first four comparisons form the "
        "pre-specified family and their $p$ values are Holm-adjusted within seed; $^e$ marks "
        "exploratory comparisons (unadjusted).",
        "tab:s_comparisons", "lcllll",
        "Comparison & Seed & $\\Delta$ malignancy F1 [95\\% CI] & $p$ & $\\Delta$ risk F1 "
        "[95\\% CI] & $p$", body[:-1], star=True, size="\\scriptsize", fit=True)


def supp_tost(n: dict) -> str:
    body = []
    for k, v in n["decoder"]["pairs"].items():
        tag, seed = k.split("_vs_0_5b_s")
        label = dict(SIZES_REV)[tag]
        cells = []
        for ep in ("pathology", "risk"):
            e = v[ep]
            mark = CHECK if e["equivalent_at_margin"] else ""
            cells.append(f"{e['delta_f1']:+.3f} [{e['ci90'][0]:+.3f}, {e['ci90'][1]:+.3f}] & "
                         f"{e['supported_margin']:.3f}{mark}")
        body.append(f"{label} vs 0.5B & {seed} & {' & '.join(cells)} \\\\")
    return table(
        "Decoder-scale equivalence (TOST, pre-specified margin $\\pm0.02$), each size against the "
        "0.5B decoder at the same seed: F1 difference with 90\\% patient-cluster bootstrap "
        "interval and the smallest symmetric margin the interval supports; $\\checkmark$ = "
        "equivalent at $\\pm0.02$.",
        "tab:s_tost", "lclclc",
        "Pair & Seed & $\\Delta$ malignancy [90\\% CI] & Margin & $\\Delta$ risk [90\\% CI] & "
        "Margin", body, star=True, size="\\scriptsize")


SIZES_REV = [(t, lbl) for lbl, t in SIZES]


def supp_interventions(n: dict) -> str:
    a = n["analyses"]["models"]
    heads = ["pathology", "risk", "birads", "orientation", "margins", "shape", "echogenicity",
             "calcification"]
    body = []
    for h in heads:
        cells = [msd(a[m]["intervention"][h]) if h in a[m]["intervention"] else "n/a"
                 for m in ("cb3", "cb9")]
        label = "BI-RADS-like category" if h == "birads" else h.capitalize()
        body.append(f"{label} & " + " & ".join(cells) + " \\\\")
    cov = [f"{mname} & {msd(a[m]['covariation']['adopt'])} & "
           f"{msd(a[m]['covariation']['flip_change'])} & "
           f"{msd(a[m]['covariation']['control_change'])} \\\\"
           for m, mname in (("cb3", "CB-3"), ("cb9", "CB-9"))]
    t1 = table(
        "Forced-concept agreement per head (80 test images, every class of the head forced in "
        "turn; fraction of all regenerated reports that state the forced value, so a report that "
        "omits the slot counts as not following the override). Mean $\\pm$ SD over "
        "three seeds; the six finding heads exist only in CB-9 (boundary has no report slot).",
        "tab:s_intervention", "lcc", "Concept head & CB-3 & CB-9", body)
    t2 = table(
        "Descriptor co-variation under a forced pathology flip (80 test images). ``Adopted'' is "
        "the fraction of flips whose report states the forced pathology; the change rates are the "
        "fraction of reports in which at least one descriptor changes, on a genuine flip and on a "
        "control that forces the pathology the unmodified report already states (itself an "
        "intervention, since it replaces the predicted distribution with a one-hot vector). A "
        "change shows that the report responds to the override, not that it is correct.",
        "tab:s_covariation", "lccc",
        "Model & Adopted & Descriptor change, flip & Descriptor change, control", cov)
    return t1 + "\n" + t2


SLOTS = ("pathology", "birads", "orientation", "margins", "shape", "echogenicity", "calcification")


def supp_extractor() -> str:
    v = json.load(open("outputs/stats/extractor_validation_v5.json"))
    body = []
    for slot in SLOTS:
        cells = []
        for style in "ABCD":
            e = v["styles"][style][slot]
            cells.append(f"{e['extraction_rate']:.3f} / {e['accuracy']:.3f}")
        body.append(f"{slot.replace('birads', 'BI-RADS-like')} & " + " & ".join(cells) + " \\\\")
    return table(
        f"Extractor validation on template-conformant text: the {v['n_records']} official-test "
        "reports that have BUS-CoT structured fields, rendered in each of the four training "
        "styles, compared with those fields (extraction rate / accuracy). "
        f"{v['n_unmatched']} test records without structured fields are not included.",
        "tab:s_extractor", "lcccc", "Slot & Style A & Style B & Style C & Style D", body)


def _min_pr(v: dict) -> float:
    return min(a[k] for e in v.values() for a in e["per_annotator"]
               for k in ("precision", "recall"))


def supp_annotation() -> str:
    """Round 2 (outputs/annotation7b) reads reports of the final evaluation; round 1
    (outputs/annotation7) read reports generated before the evaluation corrections."""
    path = Path("outputs/annotation7b/extractor_validation.json")
    first = _min_pr(json.load(open("outputs/annotation7/extractor_validation.json"))["per_slot"])
    if not path.exists():
        body = ["\\multicolumn{4}{l}{\\PENDING{two annotators fill outputs/annotation7b/"
                "sheet\\_A.html and sheet\\_B.html; then run scripts/annotation\\_sheet.py "
                "score}} \\\\"]
    else:
        v = json.load(open(path))["per_slot"]

        def both(slot: str, k: str) -> str:
            return " / ".join(f"{a[k]:.3f}" for a in v[slot]["per_annotator"])
        body = [f"{slot.replace('birads', 'BI-RADS-like')} & {both(slot, 'precision')} & "
                f"{both(slot, 'recall')} & "
                + (f"{v[slot]['inter_annotator_kappa']:.3f}"
                   if v[slot].get("inter_annotator_kappa") is not None else "n/a") + " \\\\"
                for slot in SLOTS if slot in v]
    return table(
        "Extractor validation on generated text: 100 reports of the final evaluation (20 from "
        "each of CB-3, CB-9, the 7B decoder, USFM and Qwen2.5-VL-7B, seed 1) read independently "
        "by two annotators who were blinded to the generating model and to the extractor "
        "output. Precision and recall of the extractor against annotator A / annotator B; "
        "Cohen's $\\kappa$ between the annotators. An earlier round on 100 other reports, "
        "generated before the evaluation corrections described in Section 2, gave precision and "
        f"recall of at least {first:.2f} against either annotator.",
        "tab:s_annotation", "lccc", "Slot & Precision & Recall & $\\kappa$", body)


def supp_leakage() -> str:
    leak = json.load(open("outputs/stats/leakage_v5.json"))
    h = leak["hash"]
    body = [
        f"Train / val / test records & {leak['n_records']['train']} / {leak['n_records']['val']} "
        f"/ {leak['n_records']['test']} \\\\",
        f"Patient groups train / val / test & {leak['n_studies']['train']} / "
        f"{leak['n_studies']['val']} / {leak['n_studies']['test']} \\\\",
        f"Patient groups shared train--test & {leak['train_test_study_overlap']} \\\\",
        f"Patient groups shared train--val & {leak['train_val_study_overlap']} \\\\",
        f"Near-duplicate frames test in train / test in val / val in train & "
        f"{h['splits']['test_in_train']['n']} / {h['splits']['test_in_val']['n']} / "
        f"{h['splits']['val_in_train']['n']} \\\\",
        f"U2-BENCH breast frames in train or val & {h['external']['u2bench']['in_train_val']} of "
        f"{h['external']['u2bench']['n']} (removed from external evaluation) \\\\",
        f"BrEaST frames in train or val & {h['external']['breast']['in_train_val']} of "
        f"{h['external']['breast']['n']} \\\\",
    ]
    return table(
        "Leakage audit of the revised split (scripts/leakage\\_audit.py). Groups are BUS-CoT "
        "patient identifiers joined over near-duplicate frames (256-bit difference hash, "
        "Hamming distance $\\leq10$), computed on the raw source frame behind every lesion crop.",
        "tab:s_leakage", "lr", "Check & Result", body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outdir", default="paper/frontiers/revision2/tables")
    args = ap.parse_args()
    n = json.load(open(NUM))
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    tabs = {"tab_main": tab_main(n), "tab_decoders": tab_decoders(n),
            "tab_encoders": tab_encoders(n), "tab_external": tab_external(n),
            "tab_external_reports": tab_external_reports(n), "supp_runs": supp_runs(),
            "supp_comparisons": supp_comparisons(n), "supp_tost": supp_tost(n),
            "supp_interventions": supp_interventions(n), "supp_leakage": supp_leakage(),
            "supp_extractor": supp_extractor(), "supp_annotation": supp_annotation()}
    for name, tex in tabs.items():
        header = "% Generated by scripts/revision2_tables.py; do not edit.\n"
        (out / f"{name}.tex").write_text(header + tex)
    print(f"{len(tabs)} tables -> {out}")


if __name__ == "__main__":
    main()
