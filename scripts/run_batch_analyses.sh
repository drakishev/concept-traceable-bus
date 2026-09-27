#!/usr/bin/env bash
# Downstream analyses on the concept-bottleneck checkpoints of one training batch.
#
# One stream per seed, each pinned to one GPU; every step is skipped when its
# output already exists, so the script can be re-run after an interruption.
# Outputs are tied to the checkpoint that produced them: <OUT>/checkpoint_sha256/<tag>
# holds its SHA-256, and when the checkpoint changes (a retrained model, or the
# bundled results of the public release) the tag's old outputs are moved to
# <OUT>/superseded/<time>/ and every step runs again.
# Per seed s and bottleneck model m in MODELS (cb3 = three assessment heads,
# cb9 = nine heads incl. findings), checkpoint <PREFIX><m>_dinov2_s<s>:
#   1. U2-BENCH binary malignancy and BI-RADS subset (109), head + report
#   2. BrEaST descriptor-level report evaluation (252 lesion crops)
#   3. concept-head malignant probabilities on val / test / U2-BENCH / BrEaST
#      -> calibration + sensitivity-first operating point
#   4. concept-intervention agreement (every head with a report slot) and
#      descriptor co-variation under a forced pathology flip
#   5. linear probe on frozen encoder features
#   6. Stage-2 embeddings for the modality-gap analysis
#
# U2-BENCH reuses public collections that are also in BUS-CoT (its BUSI subsets),
# so with LEAKAGE=<leakage_audit.py --hash output> the frames found in train/val
# are removed before any U2-BENCH evaluation.
#
# Usage:
#   source .venv/bin/activate && export $(grep -v '^#' .env | xargs)
#   GPUS=0,2,7 bash scripts/run_batch_analyses.sh                  # batch 6 (defaults)
#   PREFIX=h_ DATA=v5 OUT=outputs/analysis7 MODELS="cb3 cb9" \
#       LEAKAGE=outputs/stats/leakage_v5.json GPUS=5,6,5 bash scripts/run_batch_analyses.sh
set -uo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-1,2,3}"
GPUS="${GPUS:-0,2,7}"
PREFIX="${PREFIX:-g_}"
DATA="${DATA:-v4g}"
OUT="${OUT:-outputs/analysis}"
MODELS="${MODELS:-cb3}"
LEAKAGE="${LEAKAGE:-}"
U2B=data/raw/u2bench/breast_eval/breast.jsonl
BREAST=data/raw/breast/breast_eval.jsonl
TEST=data/unified_${DATA}/test_buscot_only.jsonl
VAL=data/unified_${DATA}/val.jsonl
TRAIN=data/unified_${DATA}/train.jsonl
# Stage-2 config of the models: evaluation applies its image preprocessing
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/train/finetune_jepa_${DATA}.yaml}"
mkdir -p "$OUT" logs/analysis

if [ -n "$LEAKAGE" ]; then
    python - "$LEAKAGE" "$U2B" "$OUT/u2bench_external.jsonl" <<'PY'
import json, sys
leak, src, dst = sys.argv[1:]
bad = set(json.load(open(leak))["hash"]["external"]["u2bench"]["contaminated_ids"])
rows = [line for line in open(src) if json.loads(line)["id"] not in bad]
open(dst, "w").writelines(rows)
print(f"U2-BENCH: {len(rows)} external frames kept, {len(bad)} in train/val removed")
PY
    U2B="$OUT/u2bench_external.jsonl"
fi

best_ckpt() {
    python - "$1" <<'PY'
import sys
sys.path.insert(0, "scripts")
from run_weekend import best_ckpt
print(best_ckpt(sys.argv[1]) or "")
PY
}

run_model() {  # run_model <model tag> <seed> <gpu>
    local m="$1" s="$2" gpu="$3"
    local tag="${m}_s${s}"
    local log="logs/analysis/${PREFIX}${tag}.log"
    local ck
    ck=$(best_ckpt "checkpoints/weekend/${PREFIX}${m}_dinov2_s${s}")
    if [ -z "$ck" ]; then echo "[$tag] no checkpoint yet" | tee -a "$log"; return; fi
    echo "[$tag gpu${gpu}] $ck" | tee -a "$log"
    export CUDA_VISIBLE_DEVICES="$gpu"

    local stamp="$OUT/checkpoint_sha256/$tag" sha
    sha=$(sha256sum "$ck" | cut -d' ' -f1)
    if [ "$(cat "$stamp" 2>/dev/null)" != "$sha" ]; then
        local old="$OUT/superseded/$(date +%Y%m%d-%H%M%S)" p
        for p in u2b_malig/$tag u2b_birads/$tag breast/$tag probs/$tag intervention/$tag.json \
                 covariation/$tag.json linear_probe/$tag.json viz/$tag; do
            if [ -e "$OUT/$p" ]; then
                mkdir -p "$old/$(dirname "$p")" && mv "$OUT/$p" "$old/$p"
            fi
        done
        [ -d "$old" ] && echo "[$tag] outputs of another checkpoint moved to $old" | tee -a "$log"
        mkdir -p "$(dirname "$stamp")" && echo "$sha" > "$stamp"
    fi

    step() {  # step <output-that-marks-completion> <command...>
        local marker="$1"; shift
        if [ -e "$marker" ]; then echo "[$tag] skip $marker" | tee -a "$log"; return; fi
        echo "[$tag] $(date '+%H:%M') $*" | tee -a "$log"
        "$@" >> "$log" 2>&1 || echo "[$tag] FAILED: $*" | tee -a "$log"
    }

    step "$OUT/u2b_malig/$tag/metrics.json" python scripts/evaluate_u2bench.py \
        --checkpoint "$ck" --breast_jsonl "$U2B" --output_dir "$OUT/u2b_malig/$tag" \
        --num_workers 2
    step "$OUT/u2b_birads/$tag/metrics_birads.json" python scripts/evaluate_u2bench.py \
        --checkpoint "$ck" --breast_jsonl "$U2B" --output_dir "$OUT/u2b_birads/$tag" \
        --task birads --num_workers 2

    step "$OUT/breast/$tag/predictions.json" python scripts/evaluate_jepa.py \
        --checkpoint "$ck" --test_jsonl "$BREAST" --output_dir "$OUT/breast/$tag" \
        --train_config "$TRAIN_CONFIG" --num_workers 2
    step "$OUT/breast/$tag/descriptors.json" python scripts/evaluate_breast.py \
        --predictions "$OUT/breast/$tag/predictions.json" --eval_jsonl "$BREAST" \
        --output "$OUT/breast/$tag/descriptors.json"

    local pd="$OUT/probs/$tag"; mkdir -p "$pd"
    step "$pd/val.json" python scripts/dump_concept_probs.py --checkpoint "$ck" \
        --jsonl "$VAL" --output "$pd/val.json" --num_workers 2
    step "$pd/test.json" python scripts/dump_concept_probs.py --checkpoint "$ck" \
        --jsonl "$TEST" --output "$pd/test.json" --num_workers 2
    step "$pd/u2bench.json" python scripts/dump_concept_probs.py --checkpoint "$ck" \
        --u2bench "$U2B" --output "$pd/u2bench.json" --num_workers 2
    step "$pd/breast.json" python scripts/dump_concept_probs.py --checkpoint "$ck" \
        --jsonl "$BREAST" --output "$pd/breast.json" --num_workers 2
    step "$pd/calibration.json" python scripts/calibration.py --val "$pd/val.json" \
        --test "$pd/test.json" --external "u2bench=$pd/u2bench.json" "breast=$pd/breast.json" \
        --output "$pd/calibration.json"

    step "$OUT/intervention/$tag.json" python scripts/concept_intervention.py \
        --checkpoint "$ck" --test_jsonl "$TEST" --train_config "$TRAIN_CONFIG" --n 80 \
        --output "$OUT/intervention/$tag.json"
    step "$OUT/covariation/$tag.json" python scripts/intervention_covariation.py \
        --checkpoint "$ck" --test_jsonl "$TEST" --train_config "$TRAIN_CONFIG" --n 80 \
        --output "$OUT/covariation/$tag.json"

    step "$OUT/linear_probe/$tag.json" python scripts/linear_probe.py --checkpoint "$ck" \
        --train_jsonl "$TRAIN" --test_jsonl "$TEST" \
        --restrict-to "outputs/weekend/${PREFIX}${m}_dinov2_s${s}/predictions.json" \
        --output "$OUT/linear_probe/$tag.json" --num_workers 2

    step "$OUT/viz/$tag/embeddings.npz" python scripts/visualize_embeddings.py \
        --checkpoint "$ck" --test_jsonl "$TEST" --train_config "$TRAIN_CONFIG" \
        --output_dir "$OUT/viz/$tag" --stage finetune
    echo "[$tag] done $(date '+%H:%M')" | tee -a "$log"
}

run_seed() {
    local s="$1" gpu="$2" m
    for m in $MODELS; do
        run_model "$m" "$s" "$gpu"
    done
}

IFS=',' read -ra seed_arr <<< "$SEEDS"
IFS=',' read -ra gpu_arr <<< "$GPUS"
for i in "${!seed_arr[@]}"; do
    run_seed "${seed_arr[$i]}" "${gpu_arr[$i % ${#gpu_arr[@]}]}" &
done
wait
echo "all seeds done"
