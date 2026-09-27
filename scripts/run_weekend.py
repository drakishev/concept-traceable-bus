"""Weekend experiment queue runner for the bottleneck-spectrum study.

Runs a list of jobs back-to-back, unattended, continuing past any failure.
Each job: optional Stage-1 pretrain (or reuse an existing checkpoint) -> Stage-2
finetune -> eval -> slot metrics. Results are appended to a CSV after each job, so
partial progress is never lost. Re-running skips jobs whose metrics already exist.
With EVAL_ONLY=1 nothing is trained: every job whose checkpoint exists is evaluated
again unless its outputs already record an evaluation of that checkpoint with the
training preprocessing (eval_config.json).

Usage (launch in tmux):
    export $(grep -v '^#' .env | xargs); export CUDA_VISIBLE_DEVICES=2
    python scripts/run_weekend.py 2>&1 | tee logs/weekend/runner.log
"""
from __future__ import annotations

import csv
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
LOGDIR = ROOT / "logs" / "weekend"
LOGDIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = ROOT / "outputs" / "weekend_results.csv"
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

TEST = "data/unified_v2/test_buscot_only.jsonl"
# Reusable Stage-1 checkpoint with predictor + 3 aux heads + BiomedCLIP (v2-mt)
V2MT_CKPT = "checkpoints/ultrasound_jepa_pretrain_v2_mt/epoch=09-val/loss=2.0991.ckpt"
BIOMEDCLIP_MODEL = "configs/model/ultrasound_jepa_biomedclip.yaml"
PRETRAIN_CFG = "configs/train/pretrain_jepa_v2_mt.yaml"
FINETUNE_CFG = "configs/train/finetune_jepa_v2_mt.yaml"


# ── Job definitions ────────────────────────────────────────────────────────
# type "ft": finetune-only, reuse V2MT_CKPT (fast, ~1.5h each).
# type "full": Stage-1 pretrain (new arch/data) then Stage-2 (~2.5h each).
# overrides: OmegaConf dotlist applied to the base config.
#
# Ordered: safest/highest-value first so the weekend yields results even if a
# later job stalls.

HEADS3 = "train.aux_heads=[pathology,risk,birads]"
CB = ["train.concept_bottleneck=true", HEADS3]

JOBS = [
    # ---- Group A: Stage-2-only concept-bottleneck spectrum & variants (reuse v2-mt Stage1) ----
    {"name": "cb_dim32",   "type": "ft", "ov": CB + ["train.concept_dim=32"]},
    {"name": "cb_dim128",  "type": "ft", "ov": CB + ["train.concept_dim=128"]},
    {"name": "cb_dim256",  "type": "ft", "ov": CB + ["train.concept_dim=256"]},
    {"name": "cb_hard",    "type": "ft",
     "ov": CB + ["train.concept_dim=64", "train.concept_hard=true"]},
    {"name": "cb_resid64", "type": "ft",
     "ov": CB + ["train.concept_dim=64", "train.concept_residual_dim=64"]},
    {"name": "cb_resid256", "type": "ft",
     "ov": CB + ["train.concept_dim=64", "train.concept_residual_dim=256"]},
    # ---- Group B: multi-query bottleneck (B1) — reuse v2-mt Stage1 (predictor K mismatch!) ----
    # NOTE: multi-query changes the predictor architecture, so it CANNOT reuse the
    # K=1 v2-mt checkpoint; these are "full" jobs (own Stage 1).
    {"name": "mq_k4",  "type": "full", "ov": ["model.predictor_num_query_tokens=4"]},
    {"name": "mq_k8",  "type": "full", "ov": ["model.predictor_num_query_tokens=8"]},
    {"name": "mq_k16", "type": "full", "ov": ["model.predictor_num_query_tokens=16"]},
    {"name": "mq_k32", "type": "full", "ov": ["model.predictor_num_query_tokens=32"]},
    # ---- Group C: small decoder sweep with concept bottleneck (Stage-2-only) ----
    {"name": "cb_qwen0_5b", "type": "ft",
     "ov": CB + ["model.y_decoder_name=Qwen/Qwen2.5-0.5B-Instruct"]},
    {"name": "cb_qwen1_5b", "type": "ft",
     "ov": CB + ["model.y_decoder_name=Qwen/Qwen2.5-1.5B-Instruct"]},
    # ---- Group D: X-encoder swap (full pretrain) — DINOv2 ----
    {"name": "x_dinov2", "type": "full",
     "ov": ["model.x_encoder_name=vit_large_patch14_dinov2.lvd142m",
            "model.x_encoder_embed_dim=1024"]},
]


def run(cmd: str, logfile: Path, timeout: int = 36000, cuda: str | None = None) -> int:
    """Run a shell command, tee output to logfile, return exit code.

    If `cuda` is given, the subprocess is pinned to that GPU via an inline env
    assignment (race-free across parallel workers — does not touch os.environ).
    """
    if cuda is not None:
        cmd = f"CUDA_VISIBLE_DEVICES={cuda} {cmd}"
    with open(logfile, "a") as f:
        f.write(f"\n\n===== CMD: {cmd}\n")
        f.flush()
        p = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
        return p.returncode


def best_ckpt(output_dir: str) -> str | None:
    """Find the lowest val/loss checkpoint under a training output dir.

    Skips checkpoints that are not readable zip archives — a checkpoint saved
    while the node froze/was killed mid-write is truncated and would crash
    torch.load with a 'failed finding central directory' error.
    """
    import zipfile
    cks = [c for c in glob.glob(f"{output_dir}/epoch=*-val/loss=*.ckpt")
           if zipfile.is_zipfile(c)]
    if not cks:
        f = f"{output_dir}/final.ckpt"
        return f if (os.path.exists(f) and zipfile.is_zipfile(f)) else None
    # filename embeds loss=X.XXXX.ckpt -> pick min
    def loss_of(p):
        try:
            return float(p.split("loss=")[-1].replace(".ckpt", ""))
        except Exception:
            return 1e9
    return min(cks, key=loss_of)


def metrics_for(eval_dir: str) -> dict:
    out = {}
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


def append_result(row: dict) -> None:
    exists = RESULTS_CSV.exists()
    cols = ["name", "status", "path_f1", "risk_f1", "bleu4", "meteor", "rougeL", "ckpt", "seconds"]
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k) for k in cols})


def evaluated_as_trained(eval_dir: str, ck: str) -> bool:
    """True if eval_dir holds slot metrics of checkpoint `ck` generated with the
    preprocessing of the config it was trained with."""
    from src.data.datasets.bus_cot_jepa import preprocess_mode_of
    cfg = Path(eval_dir, "eval_config.json")
    if not (cfg.exists() and Path(eval_dir, "slot_metrics.json").exists()):
        return False
    e = json.loads(cfg.read_text())
    return (e.get("checkpoint") == ck
            and e.get("preprocess_mode") == preprocess_mode_of(FINETUNE_CFG))


def evaluate(name: str, ck: str, log: Path, tmo: int, cuda: str | None, t0: float) -> dict:
    """Generate the test reports of checkpoint `ck` and score their clinical slots."""
    py, eval_dir = sys.executable, f"outputs/weekend/{name}"
    print(f"[{name}] eval ...")
    rc = run(f"{py} scripts/evaluate_jepa.py --checkpoint '{ck}' --output_dir {eval_dir} "
             f"--test_jsonl {TEST} --train_config {FINETUNE_CFG}", log, timeout=tmo, cuda=cuda)
    if rc != 0:
        return {"name": name, "status": f"eval_fail_rc{rc}", "ckpt": ck,
                "seconds": int(time.time()-t0)}
    run(f"{py} scripts/evaluate_slots.py --predictions {eval_dir}/predictions.json "
        f"--output_dir {eval_dir}", log, cuda=cuda)
    return {"name": name, "status": "ok", "ckpt": ck, "seconds": int(time.time()-t0),
            **metrics_for(eval_dir)}


def do_job(job: dict) -> dict:
    name = job["name"]
    log = LOGDIR / f"{name}.log"
    eval_dir = f"outputs/weekend/{name}"
    t0 = time.time()

    py = sys.executable
    ft_out = f"checkpoints/weekend/{name}"
    cuda = job.get("cuda")  # GPU to pin (set by the parallel pool runner)
    # Per-job wall-clock cap. The 10 h default is fine for a ~2 h run but the
    # 27B/70B decoders need longer; a job may raise it via {"timeout": seconds}.
    tmo = int(job.get("timeout", 36000))
    if os.environ.get("EVAL_ONLY"):
        ck = best_ckpt(ft_out)
        if ck is None:
            return {"name": name, "status": "no_ckpt", "seconds": 0}
        if evaluated_as_trained(eval_dir, ck):
            print(f"[SKIP] {name} (already evaluated with the training preprocessing)")
            return {"name": name, "status": "skip", **metrics_for(eval_dir)}
        return evaluate(name, ck, log, tmo, cuda, t0)
    # A job is finished only if its checkpoint exists too: metrics alone can be
    # bundled results (public release) with nothing trained on this machine.
    if os.path.exists(f"{eval_dir}/slot_metrics.json") and best_ckpt(ft_out) is not None:
        print(f"[SKIP] {name} (already has metrics and a checkpoint)")
        return {"name": name, "status": "skip", **metrics_for(eval_dir)}

    try:
        if job["type"] == "full":
            pre_out = f"checkpoints/weekend/{name}_pre"
            pre_ckpt = best_ckpt(pre_out)
            if pre_ckpt is None:
                ov = " ".join(job["ov"]) + f" train.output_dir={pre_out}"
                print(f"[{name}] Stage 1 pretrain ...")
                rc = run(f"{py} scripts/train_jepa.py --config {PRETRAIN_CFG} {ov}",
                         log, timeout=tmo, cuda=cuda)
                if rc != 0:
                    return {"name": name, "status": f"pretrain_fail_rc{rc}",
                            "seconds": int(time.time()-t0)}
                pre_ckpt = best_ckpt(pre_out)
                # Marker for jobs that reuse this Stage 1 (a checkpoint file alone
                # can be an intermediate epoch of a run that is still training).
                Path(pre_out, "DONE").touch()
            stage1 = pre_ckpt
        else:
            # ft job: reuse an existing Stage-1 checkpoint (per-job override or v2-mt)
            stage1 = job.get("pretrain_ckpt", V2MT_CKPT)

        # Stage 2 finetune
        ov = (" ".join(job["ov"])
              + f" train.output_dir={ft_out} train.pretrain_checkpoint='{stage1}'")
        print(f"[{name}] Stage 2 finetune ...")
        rc = run(f"{py} scripts/train_jepa.py --config {FINETUNE_CFG} {ov}",
                 log, timeout=tmo, cuda=cuda)

        # Resilience: on a shared node the finetune process is occasionally
        # SIGKILLed (rc137) AFTER it has already trained + saved a good checkpoint.
        # If a checkpoint survived, salvage it by proceeding to eval rather than
        # discarding the whole job.
        ck = best_ckpt(ft_out)
        if rc != 0 and ck is None:
            return {"name": name, "status": f"finetune_fail_rc{rc}", "seconds": int(time.time()-t0)}
        if rc != 0 and ck is not None:
            print(f"[{name}] finetune rc{rc} but checkpoint exists -> salvaging via eval")
        if ck is None:
            return {"name": name, "status": "no_ckpt", "seconds": int(time.time()-t0)}

        return evaluate(name, ck, log, tmo, cuda, t0)
    except Exception as e:  # never let one job kill the queue
        return {"name": name, "status": f"exception:{type(e).__name__}",
                "seconds": int(time.time()-t0)}


def main() -> None:
    print(f"Weekend runner: {len(JOBS)} jobs. Results -> {RESULTS_CSV}")
    for i, job in enumerate(JOBS, 1):
        print(f"\n===== [{i}/{len(JOBS)}] {job['name']} ({job['type']}) =====", flush=True)
        row = do_job(job)
        append_result(row)
        print(f"[DONE] {job['name']}: {row.get('status')} "
              f"path_f1={row.get('path_f1')} risk_f1={row.get('risk_f1')} "
              f"({row.get('seconds')}s)", flush=True)
    print("\nAll jobs attempted. See", RESULTS_CSV)


if __name__ == "__main__":
    main()
