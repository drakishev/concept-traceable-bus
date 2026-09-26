"""Generate the supplementary training-configuration table from the configs that ran.

Reviewer 3 item 1 asked for base learning rates, optimizer, scheduler, batch
size, epochs, model-selection criterion, quantization and decoding parameters.
Rather than transcribe them by hand (and let them drift), this script reads the
YAML configs, the batch-7 job definitions and the VLM baseline runner and emits
LaTeX for the supplement, plus a JSON copy for the release repository.

Usage:
    python scripts/make_config_table.py --out paper/frontiers/revision2/tab_config.tex
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

PRETRAIN_CFG = "configs/train/pretrain_jepa_v5.yaml"
FINETUNE_CFG = "configs/train/finetune_jepa_v5.yaml"
MODEL_CFG = "configs/model/ultrasound_jepa_biomedclip.yaml"
VLM_CFG = "configs/train/lora_sft.yaml"
VLM_MODEL_CFG = "configs/model/qwen2vl_7b.yaml"
#: Decoding parameters used by scripts/evaluate_jepa.py and evaluate_qwen.py.
DECODING = {"num_beams": 4, "max_new_tokens": 256}
#: Report-level names of the concept heads (the code uses the keys).
HEAD_NAMES = {"birads": "BI-RADS-like category", "risk": "risk group"}


def _tex(s: object) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def stage_rows(cfg, stage: str) -> list[tuple[str, str]]:
    """Plain-text rows (LaTeX escaping happens once, at emission)."""
    t = cfg.train
    return [
        (f"{stage}: optimizer", "AdamW"),
        (f"{stage}: base learning rate", f"{t.lr:g}"),
        (f"{stage}: weight decay", f"{t.weight_decay:g}"),
        (f"{stage}: betas", f"({t.betas[0]}, {t.betas[1]})"),
        (f"{stage}: schedule",
         f"cosine to {t.min_lr_ratio:g}x base, {t.warmup_steps} warm-up steps"),
        (f"{stage}: epochs",
         f"at most {t.num_epochs}; early stopping on {t.monitor}, patience "
         f"{t.early_stop_patience} (the schedule length is fixed by the epoch cap)"),
        (f"{stage}: batch size", f"{t.batch_size} (no gradient accumulation, 1 GPU)"),
        (f"{stage}: precision", str(t.precision)),
        (f"{stage}: gradient clipping", f"{t.gradient_clip_val:g}"),
        (f"{stage}: model selection",
         f"lowest {t.monitor} at epoch end (best checkpoint kept, save_top_k=1)"),
    ]


def _heads(spec: str) -> str:
    return ", ".join(HEAD_NAMES.get(h, h) for h in spec.strip("[]").split(","))


def _sweep_sizes(qwen: list) -> str:
    parts = []
    for tag, model_id, extra, _, _ in qwen:
        note = f" [{', '.join(e.split('train.')[-1] for e in extra)}]" if extra else ""
        parts.append(f"{tag.replace('_', '.')}: {model_id.split('/')[-1]}{note}")
    return ", ".join(parts)


def build_rows() -> list[tuple[str, str]]:
    pre = OmegaConf.load(ROOT / PRETRAIN_CFG)
    ft = OmegaConf.load(ROOT / FINETUNE_CFG)
    model = OmegaConf.load(ROOT / MODEL_CFG).model
    vlm = OmegaConf.load(ROOT / VLM_CFG)
    vlm_model = OmegaConf.load(ROOT / VLM_MODEL_CFG)
    import run_vlm_baselines as vb
    import run_weekend6 as b6

    lora = ", ".join(k.split(".")[-1].replace("y_decoder_", "") + "=" + v
                     for k, v in (o.split("=", 1) for o in b6.LORA_MATCHED))
    vlm_targets = ", ".join(vlm_model.lora.target_modules)
    return [
        ("Data", "BUS-CoT only; train/val from data/augmented_v5 (patient-grouped, "
                 "near-duplicate frames joined); test = BUS-CoT official test split"),
        ("Image preprocessing",
         f"{pre.data.preprocess_mode} (CLAHE + border crop), {pre.data.image_size} px; "
         "train-time horizontal flip, +/-10 degree rotation, brightness/contrast jitter"),
        ("Shared embedding dim", str(model.shared_embed_dim)),
        ("Image encoder", "frozen; see the encoder table for each backbone (batch-7 runs "
                          "add DINOv2 ViT-B/14 as the size-matched comparator for USFM)"),
        ("Predictor",
         f"{model.predictor_num_layers} transformer layers, width {model.predictor_hidden_dim}, "
         f"{model.predictor_num_heads} heads, MLP ratio {model.predictor_mlp_ratio}, "
         f"dropout {model.predictor_dropout}"),
        ("Text encoder",
         f"{model.y_encoder_name} (trainable, LR multiplier {model.y_encoder_lr_multiplier})"),
        ("Alignment objective",
         f"bidirectional InfoNCE, temperature {pre.train.temperature} "
         f"({'learnable' if pre.train.learnable_temperature else 'fixed'})"),
        ("Auxiliary concept heads",
         f"{', '.join(HEAD_NAMES.get(h, h) for h in pre.train.aux_heads)}; "
         f"loss weight {pre.train.aux_loss_weight}"),
        ("Concept bottleneck", "concept_dim 64; CB-3 heads: " + _heads(b6.HEADS3)
         + "; CB-9 heads: " + _heads(b6.HEADS9)),
        ("Decoder (encoder and bottleneck runs)",
         f"{model.y_decoder_name} (2.7B), full fine-tuning in Stage 2, "
         f"{model.y_decoder_num_soft_prompt_tokens} soft-prompt tokens, "
         f"max length {model.y_decoder_max_length}"),
        ("Decoder sweep anchor",
         "every sweep run starts Stage 2 from the Stage-1 checkpoint of CB-3 seed 1, so "
         "encoder, predictor, alignment and data are identical across sizes"),
        *stage_rows(pre, "Stage 1"),
        *stage_rows(ft, "Stage 2"),
        ("Decoder sweep LoRA", lora),
        ("Decoder sweep sizes", _sweep_sizes(b6.QWEN)),
        ("Decoder sweep quantization",
         "bf16 for 0.5B-32B; 72B: 4-bit NF4 base with double quantization, bf16 compute "
         "(QLoRA), gradient checkpointing"),
        ("Seeds", f"{list(b6.SEEDS)} for all runs except the 14B/32B/72B decoders (seed 1)"),
        ("Decoding",
         f"beam search, {DECODING['num_beams']} beams, max {DECODING['max_new_tokens']} "
         "new tokens, no sampling; generation stops at the end-of-sequence or the padding "
         "token (the token that closes every training target)"),
        ("VLM reference baselines",
         f"LoRA r={vlm_model.lora.rank}, alpha={vlm_model.lora.alpha}, "
         f"dropout {vlm_model.lora.dropout} on {vlm_targets}; lr {vb.LR:g} cosine, "
         f"warm-up {vlm.train.warmup_steps}, weight decay {vlm.train.weight_decay}, bf16, "
         f"gradient checkpointing; {vb.EPOCHS} epochs, per-device batch "
         + ", ".join(f"{j['name'].split('_', 1)[1]} {j['bs']}" for j in vb.JOBS[:3])
         + f" with {vb.GRAD_ACCUM}-step accumulation; best checkpoint by validation loss; "
         "seeds 42 (trainer default), 2 and 3, which fixed data order and dropout but not "
         "adapter initialization; greedy decoding, max 256 new tokens"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/frontiers/revision2/tab_config.tex")
    ap.add_argument("--json_out", default="outputs/stats/config_table.json")
    args = ap.parse_args()

    rows = build_rows()
    lines = [
        "% Generated by scripts/make_config_table.py from the YAML configs and the",
        "% batch-7 job definitions. Do not edit by hand.",
        # longtable: 37 rows of wrapped text do not fit on one page
        "{\\small",
        "\\begin{longtable}{>{\\raggedright\\arraybackslash}p{4.2cm}"
        ">{\\raggedright\\arraybackslash}p{11.5cm}}",
        "\\caption{Training and decoding configuration of every run reported in this paper.}",
        "\\label{tab:config}\\\\",
        "\\toprule", "Setting & Value \\\\", "\\midrule", "\\endfirsthead",
        "\\toprule", "Setting & Value \\\\", "\\midrule", "\\endhead",
    ]
    for k, v in rows:
        lines.append(f"{_tex(k)} & {_tex(v)} \\\\")
    lines += ["\\bottomrule", "\\end{longtable}", "}", ""]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_out).write_text(json.dumps(dict(rows), indent=2))
    print(f"{len(rows)} rows -> {args.out} and {args.json_out}")


if __name__ == "__main__":
    main()
