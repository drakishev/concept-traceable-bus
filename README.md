# Concept-traceable breast ultrasound report generation

Code, split definitions and results for the article
"Concept-traceable breast ultrasound report generation: encoder suitability over decoder scale"
(Rakishev, Abdikenov, Saidnassim, Orazayev, Ayanbayev; Frontiers in Medicine, under revision).

A frozen image encoder and a trainable predictor are aligned to a text encoder with bidirectional InfoNCE (Stage 1).
A decoder then writes the report conditioned only on predicted clinical concepts (Stage 2): three assessment concepts (CB-3) or those plus six BI-RADS lexicon findings (CB-9).
The repository contains everything needed to rebuild the leakage-free split, retrain every model and regenerate every number, table and figure in the article.

## Contents

| Path | What it is |
|---|---|
| `src/` | Model (encoders, predictor, concept bottleneck, decoders), data loaders, split builder, perceptual-hash leakage tools, slot extractor and metrics |
| `scripts/download/` | Download and preparation of BUS-CoT, U2-BENCH (breast subset) and BrEaST |
| `scripts/preprocess/` | Build the patient- and frame-disjoint split and the four-style training reports |
| `scripts/train_jepa.py`, `scripts/train.py` | Training of the concept-bottleneck models and of the VLM reference baselines (the file name is historical; the method is not a JEPA) |
| `scripts/run_weekend7.py`, `scripts/run_vlm_baselines.py` | The job lists and GPU pool that produced every reported run |
| `scripts/run_batch_analyses.sh` | External evaluation, calibration, interventions, linear probe and embeddings |
| `scripts/stats.py`, `scripts/decoder_tost.py`, `scripts/leakage_audit.py` | Statistics and the leakage audit |
| `scripts/revision2_numbers.py`, `scripts/revision2_tables.py`, `scripts/make_revision2_figures.py` | Every number, table and figure of the article, computed from the outputs |
| `scripts/check_manuscript.py` | Checks that every number quoted in the manuscript text appears in the computed results |
| `configs/` | Model and training configurations of the reported runs |
| `splits/` | Record identifiers, patient groups and split of every record; official-test ids of the two earlier splits; U2-BENCH frames excluded from external evaluation |
| `outputs/` | Per-run predictions and metrics (`outputs/weekend/`), statistics (`outputs/stats/`), analysis outputs (`outputs/analysis7/`) and the extractor annotation study (`outputs/annotation7/`) behind the article, in the layout the scripts read |
| `checkpoints/SHA256SUMS` | SHA-256 hashes of the 57 reported checkpoints |
| `tests/` | Unit tests of the data loaders and split builder (`pytest tests/`) |

## Installation

Python 3.10 and a CUDA GPU are required; the reported runs used NVIDIA H200 GPUs.

```bash
git clone https://github.com/drakishev/concept-traceable-bus.git && cd concept-traceable-bus
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,download]"
# exact versions of the reported runs:
pip install -r requirements-lock.txt
```

Set `OMP_NUM_THREADS=8` (the runners do this) when several jobs share a machine.
Without it every PyTorch process starts one thread per core and model construction stalls.

## Data

No images are redistributed here; all three datasets are public and must be obtained from their authors.

| Dataset | Use | Source and preparation |
|---|---|---|
| BUS-CoT | training and internal test (official test split) | Figshare, article 30838715; `python scripts/download/download_bus_cot.py` |
| U2-BENCH | external evaluation (breast subset) | Hugging Face `DolphinAI/u2-bench`; `python scripts/download/_u2b_fetch.py` writes the 690 breast frames to `data/raw/u2bench/breast_eval/breast.jsonl` |
| BrEaST | external evaluation with BI-RADS lexicon annotation | TCIA, CC BY 4.0, https://doi.org/10.7937/9WKK-Q141; download the PNG zip and the clinical XLSX by hand (Data Usage Agreement), then `python scripts/download/download_breast.py` writes `data/raw/breast/breast_eval.jsonl` |

The USFM encoder needs the authors' checkpoint `USFM_latest.pth` (from https://github.com/openmedlab/USFM) at `checkpoints/external/usfm/USFM_latest.pth`; the file used here has SHA-256 `d5fdab3edd140e4ca61471bb4087f91cd7ff2ce270db71b9cab30feda881bd17`.
All other encoders and decoders are downloaded from Hugging Face or timm on first use.

`outputs/` contains derived per-case records of the two CC BY 4.0 datasets: the BUS-CoT reference report and labels of every test case next to each generated report (`outputs/weekend/*/predictions.json`, `outputs/analysis7/probs/*/{test,val}.json`) and the BrEaST labels (`outputs/analysis7/probs/*/breast.json`, `outputs/analysis7/breast/`).
U2-BENCH is licensed CC BY-NC-ND 4.0, so for its frames the release contains only record identifiers and our own predictions, without the dataset's labels (removed with `python scripts/u2bench_labels.py strip`).
Before regenerating any number that uses U2-BENCH, download it and put the labels back:

```bash
python scripts/download/_u2b_fetch.py
python scripts/u2bench_labels.py restore
```

`scripts/revision2_numbers.py` and `scripts/make_revision2_figures.py` stop with this instruction if the labels are missing.

BUS-CoT aggregates eleven public collections, including all of BUS-BRA and BUSI.
Do not add BUS-BRA or BUSI as separate training sources: the same frames would then appear in training and test under different identifiers.

## Reproducing the article

### 1. Split

```bash
python scripts/preprocess/build_unified.py --sources bus_cot --grouped --drop_histopathology \
    --out_dir data/unified_v5
python scripts/preprocess/build_augmented.py --unified_dir data/unified_v5 \
    --lesion_json data/raw/bus_cot/BUSCoT/DatasetFiles/lesion_dataset.json \
    --out_dir data/augmented_v5
```

The split groups BUS-CoT records by patient and joins groups whose raw frames are near-duplicates (256-bit difference hash, Hamming distance at most 10).
The resulting assignment of every record is in `splits/split_manifest_v5.json`; compare your `data/unified_v5/split_manifest.json` with it.
Audit the split and the external sets:

```bash
python scripts/leakage_audit.py --data data/unified_v5 --hash \
    --external u2bench=data/raw/u2bench/breast_eval/breast.jsonl \
    "breast=data/raw/breast/BrEaST-Lesions_USG-images_and_masks/case???.png" \
    --output outputs/stats/leakage_v5.json
```

### 2. Training

Every script reads and writes one results file, `outputs/weekend7_results.csv`.
The bundled copy lists every reported run as finished, so move it aside once before retraining (not when resuming an interrupted run):

```bash
mkdir -p outputs/published && mv outputs/weekend7_results.csv outputs/published/
```

```bash
export RUNNER=scripts/run_weekend7.py RESULTS_CSV=outputs/weekend7_results.csv SAVE_TOP_K=1 NUM_WORKERS=6
# Stage-1 anchor of the decoder sweep, then the 32B and 14B decoders, on one GPU
TAG=a POOL_GPUS=3 JOB_ONLY=h_cb3_dinov2_s1,h_dec_qwen32b_s1,h_dec_qwen14b_s1 bash scripts/weekend_watchdog.sh
# the 72B decoder (4-bit) on its own GPU
TAG=b POOL_GPUS=4 JOB_ONLY=h_dec_qwen72b_s1 bash scripts/weekend_watchdog.sh
# everything else, two jobs per GPU
TAG=main POOL_GPUS=0,0,1,1,2,2 JOB_SKIP=h_cb3_dinov2_s1,h_dec_qwen32b_s1,h_dec_qwen14b_s1,h_dec_qwen72b_s1 \
    bash scripts/weekend_watchdog.sh
# VLM reference baselines, seeds 42 (trainer default), 2 and 3
VLM_DATA=v5 VLM_SEEDS=1,2,3 VLM_GPUS=5,6 \
    VLM_ONLY=vlm_qwen25vl_7b,vlm_qwen2vl_7b,vlm_internvl3_8b python scripts/run_vlm_baselines.py
```

Qwen2.5-Instruct decoders end their reports with the padding token, not with their end-of-sequence token, so generation stops on either (`src/model/y_decoder.py`); the decoder-sweep reports in `outputs/` were generated this way.
Each job trains Stage 1 and Stage 2 with early stopping on validation loss, keeps the best checkpoint, generates reports for the 873 official test records and writes `outputs/weekend/<run>/predictions.json`, replacing the bundled file (restore it with `git checkout outputs/`).
A job (including a VLM baseline) is skipped only when both its metrics and its trained checkpoint or adapter exist, so bundled metrics never stand in for a model that was not trained here.
The results file then lists only the runs trained on your machine; statistics and tables are computed from it.
The hashes of the checkpoints behind the article are in `checkpoints/SHA256SUMS`.

### 3. Analyses, statistics, tables and figures

Each analysis output is stamped with the SHA-256 of the checkpoint that produced it (`outputs/analysis7/checkpoint_sha256/`).
For a retrained checkpoint the bundled outputs of that model are moved to `outputs/analysis7/superseded/` and recomputed; for the same checkpoint an interrupted run resumes where it stopped.

```bash
PREFIX=h_ DATA=v5 OUT=outputs/analysis7 LEAKAGE=outputs/stats/leakage_v5.json \
    MODELS=cb3 GPUS=0,1,2 bash scripts/run_batch_analyses.sh
PREFIX=h_ DATA=v5 OUT=outputs/analysis7 LEAKAGE=outputs/stats/leakage_v5.json \
    MODELS=cb9 GPUS=3,4,5 bash scripts/run_batch_analyses.sh
python scripts/stats.py --runs $(ls outputs/weekend | grep '^h_') \
    --groups_jsonl data/unified_v5/test_buscot_only.jsonl --out outputs/stats/stats_batch7.json \
    --compare $(cat configs/comparisons_batch7.txt)
python scripts/decoder_tost.py --prefix h_ --groups_jsonl data/unified_v5/test_buscot_only.jsonl \
    --output outputs/stats/tost_decoder_scale_batch7.json
python scripts/batch_table.py --prefix h_ --results outputs/weekend7_results.csv \
    --stats outputs/stats/stats_batch7.json --json_out outputs/stats/batch7_table.json
python scripts/revision2_numbers.py
python scripts/revision2_tables.py --outdir paper/tables
python scripts/make_revision2_figures.py --outdir paper
python scripts/make_config_table.py --out paper/tab_config.tex
```

The extractor validation on generated text (Supplementary Table S4) is scored from the two annotators' sheets:

```bash
python scripts/annotation_sheet.py score --key outputs/annotation7/sheet_key.json \
    --sheets outputs/annotation7/sheet_A.csv outputs/annotation7/sheet_B.csv \
    --out outputs/annotation7/extractor_validation.json
```

`outputs/weekend/cb_desc9_s{1,2,3}/` and `outputs/weekend_results.csv` are the nine-concept runs of the originally submitted version, kept only to re-score the seed excluded there (Supplementary Section 2).

`outputs/` holds the outputs of these steps for the reported runs, so the statistics, tables and figures can be regenerated without retraining; rebuild the split files under `data/unified_v5/` first (step 1) and restore the U2-BENCH labels (section Data).

## Evaluation conventions

Clinical endpoints are read from the generated report by `src/evaluation/slots.py`, which covers the four report styles used in training.
A run is excluded if extraction coverage on either endpoint is below 0.50 and flagged if below 0.90.
Confidence intervals resample patients, not records.
The predicted categories are BI-RADS-like labels for a single static frame, not BI-RADS assessments.

## Intended use

Research code for methodological study.
It is not a medical device, not clinical decision support, and must not be used to guide patient care.

## License

Code: MIT License (see `LICENSE`).
The MIT License applies to the code only, not to the data-derived files under `outputs/`.
Datasets remain under their original licenses: BUS-CoT (Figshare 30838715) and BrEaST (TCIA, DOI 10.7937/9WKK-Q141) are CC BY 4.0, and the BUS-CoT reference report text in `outputs/weekend/*/predictions.json` is redistributed under that license with attribution to its authors; U2-BENCH is CC BY-NC-ND 4.0 and none of its content is redistributed here.
