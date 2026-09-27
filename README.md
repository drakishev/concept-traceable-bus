# Concept-traceable breast ultrasound report generation

Code, split definitions and results for the article
"Concept-traceable breast ultrasound report generation: a leakage-controlled evaluation of accuracy and faithfulness"
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
| `scripts/train_report_model.py`, `scripts/train.py` | Training of the concept-bottleneck models and of the VLM reference baselines |
| `scripts/run_experiments.py`, `scripts/run_vlm_baselines.py` | The job lists and GPU pool that produced every reported run |
| `scripts/run_analyses.sh` | External evaluation, calibration, interventions, linear probe and embeddings |
| `scripts/stats.py`, `scripts/decoder_tost.py`, `scripts/leakage_audit.py` | Statistics and the leakage audit |
| `scripts/article_numbers.py`, `scripts/article_tables.py`, `scripts/article_figures.py` | Every number, table and figure of the article, computed from the outputs |
| `scripts/check_manuscript.py` | Checks that every number quoted in the manuscript text appears in the computed results |
| `configs/` | Model and training configurations of the reported runs |
| `splits/` | Record identifiers, patient groups and split of every record; official-test ids of the two earlier splits; U2-BENCH frames excluded from external evaluation |
| `outputs/` | Per-run predictions and metrics (`outputs/runs/`), statistics (`outputs/stats/`), analysis outputs (`outputs/analyses/`), the extractor annotation study (`outputs/extractor_validation/round2/`, and the earlier round `outputs/extractor_validation/round1/`) and the re-evaluation of the submitted version's encoder comparison (`outputs/submitted_version/preprocessing_check/`) behind the article, in the layout the scripts read |
| `checkpoints/SHA256SUMS` | SHA-256 hashes of the 57 reported checkpoints |
| `tests/` | Unit tests of the data loaders, split builder and evaluation protocol (`pytest tests/`) |

Run names have the form `<arm>_<configuration>_seed<k>`, for example `cb3_dinov2_seed1` or `dec_qwen7b_seed2`.
File and run names were normalized for this release from the names used during the study; the SHA-256 hashes in `checkpoints/SHA256SUMS` identify the checkpoint files behind every reported run.

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

`outputs/` contains derived per-case records of the two CC BY 4.0 datasets: the BUS-CoT reference report and labels of every test case next to each generated report (`outputs/runs/*/predictions.json`, `outputs/analyses/probs/*/{test,val}.json`) and the BrEaST labels (`outputs/analyses/probs/*/breast.json`, `outputs/analyses/breast/`).
U2-BENCH is licensed CC BY-NC-ND 4.0, so for its frames the release contains only record identifiers and our own predictions, without the dataset's labels (removed with `python scripts/u2bench_labels.py strip`).
Before regenerating any number that uses U2-BENCH, download it and put the labels back:

```bash
python scripts/download/_u2b_fetch.py
python scripts/u2bench_labels.py restore
```

`scripts/article_numbers.py` and `scripts/article_figures.py` stop with this instruction if the labels are missing.

BUS-CoT aggregates eleven public collections, including all of BUS-BRA and BUSI.
Do not add BUS-BRA or BUSI as separate training sources: the same frames would then appear in training and test under different identifiers.

## Reproducing the article

### 1. Split

```bash
python scripts/preprocess/build_unified.py --sources bus_cot --grouped --drop_histopathology \
    --out_dir data/split
python scripts/preprocess/build_augmented.py --unified_dir data/split \
    --lesion_json data/raw/bus_cot/BUSCoT/DatasetFiles/lesion_dataset.json \
    --out_dir data/split_augmented
```

The split groups BUS-CoT records by patient and joins groups whose raw frames are near-duplicates (256-bit difference hash, Hamming distance at most 10).
The resulting assignment of every record is in `splits/split_manifest.json`; compare your `data/split/split_manifest.json` with it.
Audit the split and the external sets:

```bash
python scripts/leakage_audit.py --data data/split --hash \
    --external u2bench=data/raw/u2bench/breast_eval/breast.jsonl \
    "breast=data/raw/breast/BrEaST-Lesions_USG-images_and_masks/case???.png" \
    --output outputs/stats/leakage_audit.json
```

### 2. Training

Every script reads and writes one results file, `outputs/runs.csv`.
The bundled copy lists every reported run as finished, so move it aside once before retraining (not when resuming an interrupted run):

```bash
mkdir -p outputs/published && mv outputs/runs.csv outputs/published/
```

```bash
export RUNNER=scripts/run_experiments.py RESULTS_CSV=outputs/runs.csv SAVE_TOP_K=1 NUM_WORKERS=6
# Stage-1 anchor of the decoder sweep, then the 32B and 14B decoders, on one GPU
TAG=a POOL_GPUS=3 JOB_ONLY=cb3_dinov2_seed1,dec_qwen32b_seed1,dec_qwen14b_seed1 bash scripts/watchdog.sh
# the 72B decoder (4-bit) on its own GPU
TAG=b POOL_GPUS=4 JOB_ONLY=dec_qwen72b_seed1 bash scripts/watchdog.sh
# everything else, two jobs per GPU
TAG=main POOL_GPUS=0,0,1,1,2,2 JOB_SKIP=cb3_dinov2_seed1,dec_qwen32b_seed1,dec_qwen14b_seed1,dec_qwen72b_seed1 \
    bash scripts/watchdog.sh
# VLM reference baselines, seeds 42 (trainer default), 2 and 3
VLM_SEEDS=1,2,3 VLM_GPUS=5,6 \
    VLM_ONLY=vlm_qwen25vl_7b,vlm_qwen2vl_7b,vlm_internvl3_8b python scripts/run_vlm_baselines.py
```

Qwen2.5-Instruct decoders end their reports with the padding token, not with their end-of-sequence token, so generation stops on either (`src/model/y_decoder.py`); the decoder-sweep reports in `outputs/` were generated this way.
Each job trains Stage 1 and Stage 2 with early stopping on validation loss, keeps the best checkpoint, generates reports for the 873 official test records and writes `outputs/runs/<run>/predictions.json`, replacing the bundled file (restore it with `git checkout outputs/`).
A job is skipped only when its `eval_config.json` records an evaluation of the checkpoint now on disk (path, size, modification time) with the training preprocessing; a finished job (Stage-2 `DONE` marker) whose evaluation is missing or stale is evaluated again without training, and a VLM baseline is skipped only when `eval_record.json` records its trained adapter file (a trained adapter without a current evaluation is evaluated again, not retrained).
Bundled metrics therefore never stand in for a model that was not trained, or was retrained, here.
The results file then lists only the runs trained on your machine; statistics and tables are computed from it.
The hashes of the checkpoints behind the article are in `checkpoints/SHA256SUMS`.

### 3. Analyses, statistics, tables and figures

Each analysis output is stamped with the SHA-256 of the checkpoint that produced it (`outputs/analyses/checkpoint_sha256/`).
For a retrained checkpoint the bundled outputs of that model are moved to `outputs/analyses/superseded/` and recomputed; for the same checkpoint an interrupted run resumes where it stopped.

```bash
LEAKAGE=outputs/stats/leakage_audit.json \
    MODELS=cb3 GPUS=0,1,2 bash scripts/run_analyses.sh
LEAKAGE=outputs/stats/leakage_audit.json \
    MODELS=cb9 GPUS=3,4,5 bash scripts/run_analyses.sh
python scripts/stats.py --runs $(ls outputs/runs) \
    --groups_jsonl data/split/test_buscot_only.jsonl --out outputs/stats/statistics.json \
    --compare $(cat configs/comparisons.txt)
python scripts/decoder_tost.py --groups_jsonl data/split/test_buscot_only.jsonl \
    --output outputs/stats/decoder_scale.json
python scripts/summary_table.py --results outputs/runs.csv \
    --stats outputs/stats/statistics.json --json_out outputs/stats/summary_table.json
python scripts/article_numbers.py
python scripts/article_tables.py --outdir paper/tables
python scripts/article_figures.py --outdir paper
python scripts/make_config_table.py --out paper/tab_config.tex
```

The extractor validation on generated text (Supplementary Table S4) is scored from the two annotators' sheets.
`annotation_sheet.py make` writes the blank sheet, the hidden key and one clickable page per annotator (`sheet_A.html`, `sheet_B.html`), which runs in any browser and downloads the filled `sheet_A.csv` / `sheet_B.csv`; the scorer refuses sheets with unanswered items or with samples or report text of another round.

```bash
python scripts/annotation_sheet.py make --runs cb3_dinov2_seed1 cb9_dinov2_seed1 dec_qwen7b_seed1 \
    enc_usfm_seed1 vlm_qwen25vl_7b_seed1 --per_run 20 --seed 1 --out outputs/extractor_validation/round2/sheet.csv
python scripts/annotation_sheet.py score --key outputs/extractor_validation/round2/sheet_key.json \
    --sheets outputs/extractor_validation/round2/sheet_A.csv outputs/extractor_validation/round2/sheet_B.csv \
    --out outputs/extractor_validation/round2/extractor_validation.json
```

`outputs/extractor_validation/round1/` is an earlier round with the same design on reports generated before the evaluation correction below.

`outputs/submitted_version/cb9_seed{1,2,3}/` and `outputs/submitted_version/runs.csv` are the nine-concept runs of the originally submitted version, kept only to re-score the seed excluded there (Supplementary Section 2).

`outputs/` holds the outputs of these steps for the reported runs, so the statistics, tables and figures can be regenerated without retraining; rebuild the split files under `data/split/` first (step 1) and restore the U2-BENCH labels (section Data).

## Evaluation preprocessing

Test and external images receive the image preprocessing of the training configuration (CLAHE and dark-border cropping, without the training augmentations).
The evaluation scripts take it from `--train_config`, have no default, and write it to `eval_config.json` next to their outputs.
An earlier version of this code built the test data with a resize-only default; the reported results were regenerated from the trained checkpoints after that was corrected (article, Supplementary Material section 2), which the runner does without training:

```bash
EVAL_ONLY=1 POOL_GPUS=0,0,1,1 python scripts/run_experiments.py
```

The concept-intervention and co-variation analyses use a seeded random sample of 80 test images (`--sample_seed 0`, the same for every model); the record identifiers are stored in each output file.

`outputs/submitted_version/preprocessing_check/` holds the submitted version's DINOv2 and UNI2-h checkpoints evaluated both ways on its 440-record test set (`scripts/evaluate_reports.py --train_config` with a resize-only or the ultrasound configuration); the resize-only run reproduces the stored predictions in `outputs/submitted_version/dinov2/` and `outputs/submitted_version/uni2h/`.

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
Datasets remain under their original licenses: BUS-CoT (Figshare 30838715) and BrEaST (TCIA, DOI 10.7937/9WKK-Q141) are CC BY 4.0, and the BUS-CoT reference report text in `outputs/runs/*/predictions.json` is redistributed under that license with attribution to its authors; U2-BENCH is CC BY-NC-ND 4.0 and none of its content is redistributed here.
