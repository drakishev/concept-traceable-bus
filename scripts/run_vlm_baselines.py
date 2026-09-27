"""End-to-end VLM reference baselines on free GPUs, in parallel.

Each job: LoRA-SFT a generic image-text-to-text VLM on augmented_v2, then eval on
the BUS-CoT test split via evaluate_qwen.py --generic + evaluate_slots.py. Results
append to outputs/weekend_results.csv (same format as the JEPA sweep) so they drop
straight into the master table.

Runs on GPUs 0,1 (the JEPA pool uses 2-7). Continue-on-error; a run is skipped only if its
metrics record the trained adapter now on disk (eval_record.json).
Gated models (MedGemma, Llama-3.2-Vision) fail fast unless the HF account has
accepted their licenses — re-run after acceptance.

Launch in tmux:
    export $(grep -v '^#' .env | xargs)
    python scripts/run_vlm_baselines.py 2>&1 | tee logs/weekend/vlm_baselines.log
"""
from __future__ import annotations

import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
# One intra-op thread per core in every process oversubscribes a shared node.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
LOGDIR = ROOT / "logs" / "weekend"
LOGDIR.mkdir(parents=True, exist_ok=True)
PY = sys.executable
SFT_CFG = "configs/train/lora_sft.yaml"
# VLM_DATA=v2  -> the published image-level split (default, historical)
# VLM_DATA=v4g -> the grouped split of revision 2 (batch 6, `g_` prefix)
# VLM_DATA=v5  -> the patient- and frame-disjoint split (batch 7, `h_` prefix)
# Job names, the results file and checkpoint dirs carry the batch prefix.
DATA = os.environ.get("VLM_DATA", "v2")
TRAIN_JSONL = f"data/augmented_{DATA}/train.jsonl"
VAL_JSONL = f"data/unified_{DATA}/val.jsonl"
TEST = f"data/unified_{DATA}/test_buscot_only.jsonl"
PREFIX, _CSV_NAME = {"v4g": ("g_", "weekend6_results.csv"),
                     "v5": ("h_", "weekend7_results.csv")}.get(DATA, ("", "weekend_results.csv"))
RESULTS_CSV = ROOT / "outputs" / _CSV_NAME
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

# per-model: hf_id + per-device batch size (smaller for the bigger / heavier ones)
JOBS = [
    {"name": "vlm_qwen25vl_7b", "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct", "bs": 2},
    {"name": "vlm_qwen2vl_7b", "hf_id": "Qwen/Qwen2-VL-7B-Instruct", "bs": 2},
    # -hf = native HF processor; InternVL tiles an image into ~3.3k tokens, so
    # raise max_seq_length to avoid truncating image placeholders (token mismatch).
    {"name": "vlm_internvl3_8b", "hf_id": "OpenGVLab/InternVL3-8B-hf", "bs": 1, "max_len": 4608},
    {"name": "vlm_medgemma_4b", "hf_id": "google/medgemma-4b-it", "bs": 2},        # gated
    {"name": "vlm_llama32_11b", "hf_id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
     "bs": 1},  # gated
]
EPOCHS = 2
GRAD_ACCUM = 8
LR = 1.0e-4


def run(cmd: str, log: Path, cuda: str) -> int:
    cmd = f"CUDA_VISIBLE_DEVICES={cuda} {cmd}"
    with open(log, "a") as f:
        f.write(f"\n\n===== CMD: {cmd}\n")
        f.flush()
        return subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT).returncode


def metrics_for(eval_dir: str) -> dict:
    out: dict = {}
    mp, sp = f"{eval_dir}/metrics.json", f"{eval_dir}/slot_metrics.json"
    if os.path.exists(mp):
        m = json.load(open(mp))
        out.update({"bleu4": m.get("bleu_4"), "meteor": m.get("meteor"),
                    "rougeL": m.get("rouge_l")})
    if os.path.exists(sp):
        s = json.load(open(sp))
        out["path_f1"] = s.get("binary_pathology", {}).get("f1")
        out["risk_f1"] = s.get("birads_risk_group", {}).get("f1")
    return out


def append_result(row: dict, lock: threading.Lock) -> None:
    cols = ["name", "status", "path_f1", "risk_f1", "bleu4", "meteor", "rougeL", "ckpt", "seconds"]
    with lock:
        exists = RESULTS_CSV.exists()
        with open(RESULTS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            if not exists:
                w.writeheader()
            w.writerow({k: row.get(k) for k in cols})


def adapter_stamp(adapter: str) -> dict:
    """Identity of a trained adapter file (a retrained adapter at the same path differs)."""
    st = Path(adapter, "adapter_model.safetensors").stat()
    return {"adapter": adapter, "adapter_size": st.st_size, "adapter_mtime_ns": st.st_mtime_ns}


def evaluated(eval_dir: str, adapter: str) -> bool:
    """True if eval_dir holds slot metrics recorded as coming from this adapter file."""
    rec = Path(eval_dir, "eval_record.json")
    return (rec.exists() and Path(eval_dir, "slot_metrics.json").exists()
            and json.loads(rec.read_text()) == adapter_stamp(adapter))


def evaluate(name: str, hf_id: str, adapter: str, eval_dir: str, log: Path, cuda: str,
             t0: float) -> dict:
    rc = run(f"{PY} scripts/evaluate_qwen.py --generic --hf_id {hf_id} --adapter {adapter} "
             f"--test_jsonl {TEST} --output_dir {eval_dir}", log, cuda)
    if rc != 0:
        return {"name": name, "status": f"eval_fail_rc{rc}", "ckpt": adapter,
                "seconds": int(time.time() - t0)}
    run(f"{PY} scripts/evaluate_slots.py --predictions {eval_dir}/predictions.json "
        f"--output_dir {eval_dir}", log, cuda)
    Path(eval_dir, "eval_record.json").write_text(json.dumps(adapter_stamp(adapter), indent=2))
    return {"name": name, "status": "ok", "ckpt": adapter, "seconds": int(time.time() - t0),
            **metrics_for(eval_dir)}


def do_job(job: dict, cuda: str) -> dict:
    # Seed 1 is the original run (trainer default seed 42, no suffix); VLM_SEEDS
    # adds replicate runs named <job>_s<seed> for a variance estimate.
    seed = job.get("seed", 1)
    suffix = f"_s{seed}" if seed != 1 else ""
    name, hf_id = PREFIX + job["name"] + suffix, job["hf_id"]
    log = LOGDIR / f"{name}.log"
    out_dir = f"checkpoints/{name}"
    eval_dir = f"outputs/weekend/{name}"
    t0 = time.time()

    adapter = f"{out_dir}/final"
    # The final adapter is written only when training ends. Metrics count only if they
    # record this adapter file: bundled results (public release) or metrics of an earlier
    # adapter are re-evaluated, and a trained adapter is never trained again.
    trained = all(os.path.exists(f"{adapter}/{f}")
                  for f in ("adapter_config.json", "adapter_model.safetensors"))
    if trained and evaluated(eval_dir, adapter):
        return {"name": name, "status": "skip", **metrics_for(eval_dir)}
    if trained:
        print(f"[{name}] evaluation missing or from another adapter: re-evaluating", flush=True)
        return evaluate(name, hf_id, adapter, eval_dir, log, cuda, t0)

    # Resume from the latest HF Trainer checkpoint if one exists (freeze recovery).
    import glob as _glob
    cks = _glob.glob(f"{out_dir}/checkpoint-*")
    resume = ""
    if cks:
        def step_of(path: str) -> int:
            tail = path.rsplit("-", 1)[-1]
            return int(tail) if tail.isdigit() else -1
        latest = max(cks, key=step_of)
        resume = f" train.resume_from_checkpoint={latest}"
        print(f"[{name}] resuming from {latest}", flush=True)

    common = (f"model.hf_id={hf_id} model.generic=true model.attn_implementation=sdpa "
              f"data.unified_jsonl={TRAIN_JSONL} data.val_jsonl={VAL_JSONL} "
              f"train.output_dir={out_dir} train.num_train_epochs={EPOCHS} "
              f"train.per_device_train_batch_size={job['bs']} "
              f"train.gradient_accumulation_steps={GRAD_ACCUM} train.lr={LR} "
              f"train.max_seq_length={job.get('max_len', 2048)}"
              + (f" train.seed={seed}" if seed != 1 else ""))
    rc = run(f"{PY} scripts/train.py --config {SFT_CFG} {common}{resume}", log, cuda)
    if rc != 0:
        return {"name": name, "status": f"train_fail_rc{rc}", "seconds": int(time.time() - t0)}

    return evaluate(name, hf_id, adapter, eval_dir, log, cuda, t0)


def worker(gpu: str, q: "queue.Queue[dict]", lock: threading.Lock) -> None:
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        print(f"[gpu{gpu}] START {job['name']} seed {job['seed']}", flush=True)
        try:
            row = do_job(job, gpu)
        except Exception as e:
            row = {"name": job["name"], "status": f"worker_exc:{type(e).__name__}"}
        append_result(row, lock)
        print(f"[gpu{gpu}] DONE  {job['name']} seed {job['seed']}: {row.get('status')} "
              f"path_f1={row.get('path_f1')} risk_f1={row.get('risk_f1')}", flush=True)


def main() -> None:
    gpus = os.environ.get("VLM_GPUS", "0,1").split(",")
    only = os.environ.get("VLM_ONLY")  # comma-separated job names to run (others skipped)
    seeds = [int(x) for x in os.environ.get("VLM_SEEDS", "1").split(",")]
    jobs = [{**j, "seed": s} for s in seeds for j in JOBS
            if only is None or j["name"] in only.split(",")]
    q: "queue.Queue[dict]" = queue.Queue()
    for j in jobs:
        q.put(j)
    print(f"VLM baselines ({DATA}): {len(jobs)} jobs across GPUs {gpus} -> {RESULTS_CSV.name}",
          flush=True)
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(g.strip(), q, lock)) for g in gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("VLM baselines done.", flush=True)


if __name__ == "__main__":
    main()
