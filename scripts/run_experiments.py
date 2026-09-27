"""Train and evaluate every reported model.

The patient- and frame-disjoint split (data/split, README step 1) is BUS-CoT only,
grouped by BUS-CoT's patient identifier, with near-duplicate frames joined into one
group and training or validation copies of test frames dropped
(scripts/leakage_audit.py --hash reports zero overlap). The jobs are those of
scripts/experiments.py plus DINOv2 ViT-B/14, the architecture-matched comparator for
USFM (ViT-B). Both stages stop early on validation loss (patience in
configs/train/stage1_alignment.yaml and configs/train/stage2_generation.yaml).

Usage (two pools: the Stage-1 anchor and the >=32B decoders own whole GPUs):
    source .venv/bin/activate && export $(grep -v '^#' .env | xargs)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    SAVE_TOP_K=1 NUM_WORKERS=6 POOL_GPUS=3,4 \\
        JOB_ONLY=cb3_dinov2_seed1,enc_uni2h_seed1,dec_qwen72b_seed1,dec_qwen32b_seed1 \\
        python scripts/run_experiments.py 2>&1 | tee logs/runs/runner_large.log
    SAVE_TOP_K=1 NUM_WORKERS=6 POOL_GPUS=0,0,1,1,2,2 \\
        JOB_SKIP=cb3_dinov2_seed1,enc_uni2h_seed1,dec_qwen72b_seed1,dec_qwen32b_seed1 \\
        python scripts/run_experiments.py 2>&1 | tee logs/runs/runner.log
    EVAL_ONLY=1 POOL_GPUS=0,0,1,1 python scripts/run_experiments.py   # evaluation only
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
import experiments as rw6  # noqa: E402
import runner as rw  # noqa: E402

rw.TEST = "data/split/test_buscot_only.jsonl"
rw.PRETRAIN_CFG = "configs/train/stage1_alignment.yaml"
rw.FINETUNE_CFG = "configs/train/stage2_generation.yaml"
rw.RESULTS_CSV = rw.ROOT / "outputs" / "runs.csv"

PREFIX = ""
JOBS = rw6.make_jobs(PREFIX) + [
    {"name": f"{PREFIX}enc_dinov2_B_seed{s}", "type": "full",
     "ov": rw6.enc("vit_base_patch14_dinov2.lvd142m", 768) + [f"train.seed={s}"]}
    for s in rw6.SEEDS
]


if __name__ == "__main__":
    rw6.main(JOBS, "Experiments")
