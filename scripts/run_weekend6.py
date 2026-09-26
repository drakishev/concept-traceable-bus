"""Batch 6 — every principal experiment retrained on the grouped split.

Revision 2 (Frontiers in Medicine, reviewers 1-3): the published runs were
trained on an image-level split, so this batch repeats them on
`data/augmented_v4g`, where the test set is BUS-CoT's own held-out split and
train/val are disjoint by source image (BUS-CoT) and by patient (BUS-BRA).
Histopathology sentences are absent from every target. See
md_files/REVISION2_PLAN.md for the decisions behind each group.

Groups (names carry the `g_` prefix so nothing collides with earlier batches):
  A  g_cb3_dinov2_s{1,2,3}   flagship: DINOv2-L + 3-assessment concept bottleneck
  B  g_cb9_dinov2_s{1,2,3}   finding-level 9-head bottleneck (reviewer 1 item 7)
  C  g_enc_<encoder>_s{1,2,3} plain model, 8 encoders incl. USFM (ultrasound) and
                              UNI2-h (modality-mismatched reference)
  D  g_dec_qwen<size>_s*      single-family Qwen2.5-Instruct decoder sweep on the
                              group-A seed-1 Stage 1, matched LoRA at every size

Differences from run_weekend5:
  * `after`: a job may name a Stage-1 directory it depends on; the worker holds
    the job (without a GPU) until that directory carries the DONE marker
    written by run_weekend.do_job.
  * `min_free_gb`: a job may demand that much free memory on the leased GPU.
    The node is shared; a 72B QLoRA run that lands on a GPU where another
    tenant holds 45 GB dies at the first step.

Usage:
    tmux new -s batch6
    source .venv/bin/activate && export $(grep -v '^#' .env | xargs)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    SAVE_TOP_K=1 NUM_WORKERS=2 POOL_GPUS=0,1,2,3,4,5,6,7 \\
        python scripts/run_weekend6.py 2>&1 | tee logs/weekend/runner6.log

    JOB_ONLY=g_cb3_dinov2_s1 POOL_GPUS=1 python scripts/run_weekend6.py   # one job
    JOB_SKIP=g_dec_qwen72b_s1 ...                                        # exclude
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_weekend as rw  # noqa: E402
from run_weekend import append_result, best_ckpt, do_job  # noqa: E402

# ── Grouped-split protocol: redirect the shared runner at the v4g data ─────
rw.TEST = "data/unified_v4g/test_buscot_only.jsonl"
rw.PRETRAIN_CFG = "configs/train/pretrain_jepa_v4g.yaml"
rw.FINETUNE_CFG = "configs/train/finetune_jepa_v4g.yaml"
rw.RESULTS_CSV = rw.ROOT / "outputs" / "weekend6_results.csv"

DINOV2 = ["model.x_encoder_name=vit_large_patch14_dinov2.lvd142m",
          "model.x_encoder_embed_dim=1024"]
HEADS3 = "[pathology,risk,birads]"
HEADS9 = "[pathology,risk,birads,margins,shape,orientation,echogenicity,boundary,calcification]"

#: Identical LoRA protocol at every decoder size.
LORA_MATCHED = ["model.y_decoder_lora_rank=16", "model.y_decoder_lora_alpha=32",
                "model.y_decoder_lora_dropout=0.05", "model.y_decoder_force_lora=true"]

SEEDS = (1, 2, 3)


def enc(name: str, dim: int) -> list[str]:
    return [f"model.x_encoder_name={name}", f"model.x_encoder_embed_dim={dim}"]


def cb(heads: str) -> list[str]:
    return ["train.concept_bottleneck=true", "train.concept_dim=64",
            f"train.aux_heads={heads}"]


ENCODERS = [
    ("dinov2_L",  "vit_large_patch14_dinov2.lvd142m",       1024),
    ("dinov3_L",  "vit_large_patch16_dinov3.lvd1689m",      1024),
    ("in21k_L",   "vit_large_patch16_224.augreg_in21k",     1024),
    ("eva02_L",   "eva02_large_patch14_224.mim_in22k",      1024),
    ("raddino",   "microsoft/rad-dino",                     768),
    ("siglip_L",  "vit_large_patch16_siglip_256.webli",     1024),
    ("uni2h",     "hf-hub:MahmoodLab/UNI2-h",               1536),
    ("usfm",      "usfm:checkpoints/external/usfm/USFM_latest.pth", 768),
]

# Same tokenizer, pretraining recipe and instruction tuning at every size, so
# only size varies. Batch size and precision are adjusted where a size would
# not fit on one H200; those deviations go in the supplementary config table.
QWEN = [
    ("0_5b", "Qwen/Qwen2.5-0.5B-Instruct", [],                     40,  SEEDS),
    ("1_5b", "Qwen/Qwen2.5-1.5B-Instruct", [],                     40,  SEEDS),
    ("3b",   "Qwen/Qwen2.5-3B-Instruct",   [],                     50,  SEEDS),
    ("7b",   "Qwen/Qwen2.5-7B-Instruct",   [],                     70,  SEEDS),
    ("14b",  "Qwen/Qwen2.5-14B-Instruct",  ["train.batch_size=8"], 100, (1,)),
    ("32b",  "Qwen/Qwen2.5-32B-Instruct",  ["train.batch_size=2"], 130, (1,)),
    # 72B: 4-bit NF4 base + LoRA (y_decoder._NEEDS_QLORA_PREFIXES), batch 2.
    ("72b",  "Qwen/Qwen2.5-72B-Instruct",  ["train.batch_size=2"], 135, (1,)),
]


def anchor_pre(prefix: str) -> str:
    """Stage 1 of the flagship seed-1 run; every decoder job reuses it (matched
    representation, alignment and data across the whole sweep)."""
    return f"checkpoints/weekend/{prefix}cb3_dinov2_s1_pre"


def make_jobs(prefix: str) -> list[dict]:
    jobs: list[dict] = []
    # A. flagship (seed 1 first: its Stage 1 is the decoder anchor)
    for s in SEEDS:
        jobs.append({"name": f"{prefix}cb3_dinov2_s{s}", "type": "full",
                     "ov": DINOV2 + cb(HEADS3) + [f"train.seed={s}"]})
    # B. finding-level concepts vs the 3-head control above
    for s in SEEDS:
        jobs.append({"name": f"{prefix}cb9_dinov2_s{s}", "type": "full",
                     "ov": DINOV2 + cb(HEADS9) + [f"train.seed={s}"]})
    # C. encoders, plain model (enc_dinov2_L is also the opaque arm of the
    #    concept-bottleneck comparison)
    for tag, model, dim in ENCODERS:
        for s in SEEDS:
            jobs.append({"name": f"{prefix}enc_{tag}_s{s}", "type": "full",
                         "ov": enc(model, dim) + [f"train.seed={s}"]})
    # D. single-family decoder sweep (Qwen2.5-Instruct, 0.5B-72B)
    for tag, model, extra, min_free, seeds in QWEN:
        for s in seeds:
            jobs.append({"name": f"{prefix}dec_qwen{tag}_s{s}", "type": "ft",
                         "after": anchor_pre(prefix), "min_free_gb": min_free,
                         "timeout": 24 * 3600 if tag in ("32b", "72b") else 36000,
                         "ov": DINOV2 + cb(HEADS3) + LORA_MATCHED + extra
                               + [f"model.y_decoder_name={model}", f"train.seed={s}"]})
    return jobs


JOBS = make_jobs("g_")


# ── Pool ──────────────────────────────────────────────────────────────────

def gpu_free_gb(gpu: str) -> float:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits", "-i", gpu],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip().split("\n")[0]) / 1024
    except ValueError:
        return 0.0


def worker(gpu_pool: "queue.Queue[str]", q: "queue.Queue[dict]", lock: threading.Lock,
           save_top_k: str | None, num_workers: str | None) -> None:
    while True:
        try:
            job = q.get(timeout=30)
        except queue.Empty:
            # Held jobs are re-queued after a sleep, so an empty queue does not
            # mean finished; `unfinished_tasks` counts held and running jobs too.
            if q.unfinished_tasks == 0:
                return
            continue
        name = job["name"]

        after = job.get("after")
        if after and not Path(after, "DONE").exists():
            q.put(job)
            q.task_done()
            time.sleep(120)
            continue
        if after:
            job["pretrain_ckpt"] = best_ckpt(after)

        cuda = gpu_pool.get()
        # Even a small job needs headroom on this shared node.
        need = job.get("min_free_gb", 40)
        free = gpu_free_gb(cuda)
        if free < need:
            print(f"[gpu{cuda}] HOLD  {name}: needs {need} GB free, has {free:.0f}", flush=True)
            gpu_pool.put(cuda)
            q.put(job)
            q.task_done()
            time.sleep(300)
            continue

        job["cuda"] = cuda
        ov = list(job["ov"])
        if save_top_k:
            ov.append(f"train.save_top_k={save_top_k}")
        if num_workers:
            ov.append(f"data.num_workers={num_workers}")
        job["ov"] = ov

        print(f"[gpu{cuda}] START {name} ({job['type']}, {free:.0f} GB free)", flush=True)
        try:
            row = do_job(job)
        except Exception as e:  # never let a worker die
            row = {"name": name, "status": f"worker_exc:{type(e).__name__}"}
        finally:
            gpu_pool.put(cuda)
        with lock:
            append_result(row)
        print(f"[gpu{cuda}] DONE  {name}: {row.get('status')} "
              f"path_f1={row.get('path_f1')} risk_f1={row.get('risk_f1')}", flush=True)
        q.task_done()


def main(all_jobs: list[dict] = JOBS, label: str = "Batch 6") -> None:
    pool = os.environ.get("POOL_GPUS", "0,1,2,3,4,5,6,7")
    gpus = [g.strip() for g in pool.split(",") if g.strip()]
    only = os.environ.get("JOB_ONLY")
    skip = set((os.environ.get("JOB_SKIP") or "").split(",")) - {""}
    save_top_k = os.environ.get("SAVE_TOP_K")
    num_workers = os.environ.get("NUM_WORKERS")

    jobs = [j for j in all_jobs
            if (only is None or j["name"] in only.split(",")) and j["name"] not in skip]

    # Queue order: the anchor first (it unblocks group D), then the independent
    # full jobs, then the decoder jobs heaviest first. Decoder jobs sit behind
    # the full jobs because they cannot start until the anchor's Stage 1 is
    # done (~1 h), and a worker that picks a blocked job re-queues it and sleeps.
    def rank(j: dict) -> tuple:
        if j["name"].endswith("cb3_dinov2_s1"):
            return (0, 0)
        if j.get("after"):
            return (2, -j.get("min_free_gb", 0))
        return (1, 0)
    jobs.sort(key=rank)

    q: "queue.Queue[dict]" = queue.Queue()
    for job in jobs:
        q.put(job)
    gpu_pool: "queue.Queue[str]" = queue.Queue()
    for g in gpus:
        gpu_pool.put(g)

    print(f"{label}: {len(jobs)} jobs across GPUs {gpus} "
          f"(save_top_k={save_top_k or 'config default'}, "
          f"num_workers={num_workers or 'config default'})", flush=True)
    print(f"  test set: {rw.TEST}\n  results:  {rw.RESULTS_CSV}", flush=True)
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(gpu_pool, q, lock, save_top_k, num_workers),
                                daemon=False) for _ in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"\n{label} done. See", rw.RESULTS_CSV, flush=True)


if __name__ == "__main__":
    main()
