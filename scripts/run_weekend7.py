"""Batch 7 — batch 6 repeated on the patient- and frame-disjoint split.

Revision 2 audit (2026-09-24): BUS-CoT aggregates 11 collections, including all
of BUS-BRA, so adding BUS-BRA as a second source put 174 of the 873 official
test frames into train/val under other ids. `data/augmented_v5` is BUS-CoT only,
grouped by BUS-CoT's patient id, with near-duplicate frames joined into one group
and train/val copies of test frames dropped (`scripts/leakage_audit.py --hash`
reports zero overlap). The test set is the same 873 records as batch 6.

Jobs are batch 6's (`h_` prefix) plus DINOv2-B, which gives USFM (ViT-B) an
architecture-matched general-purpose comparator. Both stages stop early on
val/loss (patience in configs/train/*_jepa_v5.yaml), which keeps the selected
checkpoint of every batch-6 run.

Usage (two pools: the Stage-1 anchor and the >=32B decoders own whole GPUs):
    tmux new -s batch7
    source .venv/bin/activate && export $(grep -v '^#' .env | xargs)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    SAVE_TOP_K=1 NUM_WORKERS=6 POOL_GPUS=3,4 \\
        JOB_ONLY=h_cb3_dinov2_s1,h_enc_uni2h_s1,h_dec_qwen72b_s1,h_dec_qwen32b_s1 \\
        python scripts/run_weekend7.py 2>&1 | tee logs/weekend/runner7_big.log
    SAVE_TOP_K=1 NUM_WORKERS=6 POOL_GPUS=0,0,1,1,2,2 \\
        JOB_SKIP=h_cb3_dinov2_s1,h_enc_uni2h_s1,h_dec_qwen72b_s1,h_dec_qwen32b_s1 \\
        python scripts/run_weekend7.py 2>&1 | tee logs/weekend/runner7.log
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Several jobs share the node, and each PyTorch process otherwise starts one
# intra-op thread per core (224): model construction then crawls under ~1M
# context switches/s and the GPUs idle. Jobs inherit this cap.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_weekend as rw  # noqa: E402
import run_weekend6 as rw6  # noqa: E402

rw.TEST = "data/unified_v5/test_buscot_only.jsonl"
rw.PRETRAIN_CFG = "configs/train/pretrain_jepa_v5.yaml"
rw.FINETUNE_CFG = "configs/train/finetune_jepa_v5.yaml"
rw.RESULTS_CSV = rw.ROOT / "outputs" / "weekend7_results.csv"

PREFIX = "h_"
JOBS = rw6.make_jobs(PREFIX) + [
    {"name": f"{PREFIX}enc_dinov2_B_s{s}", "type": "full",
     "ov": rw6.enc("vit_base_patch14_dinov2.lvd142m", 768) + [f"train.seed={s}"]}
    for s in rw6.SEEDS
]


if __name__ == "__main__":
    rw6.main(JOBS, "Batch 7")
