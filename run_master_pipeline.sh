#!/usr/bin/env bash
#SBATCH --job-name=emit_master
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --output=master_log_%j.txt
#SBATCH --error=master_err_%j.txt
# GC preparation and pretraining, then task-matched AP/NF five-fold fine-tuning.
# Select loss_type and masking_strategy in config/pt_config_chemo_GC.yaml.
# Usage: sbatch run_master_pipeline.sh
# Optional: set RUN_TAG, REBUILD_DATA=0, or SKIP_FORMATTING=1 as described below.
# Restricted MIMIC-derived inputs, checkpoints and results must stay private.

set -euo pipefail
export PYTHONUNBUFFERED=1

module load python
if ! command -v conda >/dev/null 2>&1; then
    echo 'conda is not available in the batch environment' >&2
    exit 1
fi
# Batch shells may not initialize the conda shell function automatically.
eval "$(conda shell.bash hook)"
conda activate emit
cd "$HOME/EMIT"

ROOT="$PWD"
RAW_DATA_DIR="$ROOT/data/raw"
MIMIC_CHEMO_DIR="$RAW_DATA_DIR/mimic_chemo"
PREPROCESSED_DIR="$ROOT/data/pre_processed"
PRETRAINING_OUTPUT_DIR="$ROOT/data/pre_training"
TOP_FEATURES_PATH="$MIMIC_CHEMO_DIR/top_features/mimic_top100_features.pkl"
AP_RANGE_FILE="$MIMIC_CHEMO_DIR/top_features/mimic_cohort_aplasia_45_days_ranges_average.csv"
NF_RANGE_FILE="$MIMIC_CHEMO_DIR/top_features/mimic_cohort_NF_30_days_ranges_average.csv"
GC_PREPROCESSED_PATH="$PREPROCESSED_DIR/chemo_GC_fold_0.pkl"
PT_CONFIG="$ROOT/config/pt_config_chemo_GC.yaml"
PT_DATA="$PRETRAINING_OUTPUT_DIR/pretraining_data_chemo_GC.npz"
PT_MASK="$PRETRAINING_OUTPUT_DIR/chemo_GC_event_masks.npz"

# Read loss and masking settings from the GC YAML used by the formatter.
require_file() {
    [[ -f "$1" ]] || {
        echo "Missing required file: $1" >&2
        exit 1
    }
}

require_file "$PT_CONFIG"

EXPERIMENT_SETTINGS="$(
    python - "$PT_CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.load(sys.argv[1])
print(cfg.loss_type)
print(cfg.masking_strategy)
PY
)"

mapfile -t SETTINGS_LINES <<< "$EXPERIMENT_SETTINGS"

[[ ${#SETTINGS_LINES[@]} -eq 2 ]] || {
    echo "Expected loss_type and masking_strategy from $PT_CONFIG" >&2
    exit 1
}

LOSS_TYPE="${SETTINGS_LINES[0]}"
MASKING_STRATEGY="${SETTINGS_LINES[1]}"

# 1: rerun GC extraction, AP/NF/GC preprocessing, and all four frequency-weight generators. 0: reuse the existing preprocessed fold and weight files.
REBUILD_DATA="${REBUILD_DATA:-1}"
# 0: rerun GC tensor formatting and generate the GC event-mask file using the current YAML. 1: reuse the existing GC tensor and mask files.
SKIP_FORMATTING="${SKIP_FORMATTING:-0}"

# Build a readable experiment label plus a fingerprint of all three YAMLs.
# RUN_TAG can still be supplied explicitly, but the default needs no CLI input.
AUTO_RUN_TAG="$(
    python - \
        "$PT_CONFIG" \
        "$ROOT/config/ft_config_chemo_AP.yaml" \
        "$ROOT/config/ft_config_chemo_NF.yaml" <<'PY'
import hashlib
import json
import re
import sys

from omegaconf import OmegaConf

pt = OmegaConf.load(sys.argv[1])
ap = OmegaConf.load(sys.argv[2])
nf = OmegaConf.load(sys.argv[3])

def window_label(cfg):
    return "7d" if bool(cfg.USE_7_DAY_WINDOW) else "14d"

def safe(text):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(text))

readable = "_".join([
    safe(pt.masking_strategy),
    safe(pt.loss_type),
    f"GC{window_label(pt)}",
    f"L{int(pt.max_len)}",
    safe(pt.target_mode),
    "hybrid" if bool(pt.USE_FORECASTING_ABLATION) else "MAE",
    f"PTbs{int(pt.batch_size)}",
    f"AP{window_label(ap)}L{int(ap.max_seq_length)}",
    f"NF{window_label(nf)}L{int(nf.max_seq_length)}",
])

configs = []
for cfg in (pt, ap, nf):
    content = OmegaConf.to_container(cfg, resolve=False)
    content.pop("hydra", None)
    configs.append(content)

payload = json.dumps(
    configs,
    sort_keys=True,
    separators=(",", ":"),
    default=str,
).encode("utf-8")
fingerprint = hashlib.sha256(payload).hexdigest()[:10]

print(f"{readable}_{fingerprint}")
PY
)"

RUN_TAG="${RUN_TAG:-$AUTO_RUN_TAG}"

case "$LOSS_TYPE" in
    standard|inverse_frequency|proportional_frequency|temporal_weighted|relative_temporal|clinical_regularized) ;;
    *) echo "Unsupported LOSS_TYPE: $LOSS_TYPE" >&2; exit 2 ;;
esac
case "$MASKING_STRATEGY" in
    rate_of_change|random|clinical|frequency|temporal) ;;
    *) echo "Unsupported MASKING_STRATEGY: $MASKING_STRATEGY" >&2; exit 2 ;;
esac
case "$REBUILD_DATA:$SKIP_FORMATTING" in
    0:0|0:1|1:0|1:1) ;;
    *) echo 'REBUILD_DATA and SKIP_FORMATTING must be 0 or 1' >&2; exit 2 ;;
esac
if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo 'RUN_TAG must contain only letters, numbers, underscores or hyphens' >&2
    exit 2
fi

for file in "$PT_CONFIG" "$ROOT/config/ft_config_chemo_AP.yaml" \
    "$ROOT/config/ft_config_chemo_NF.yaml" "$TOP_FEATURES_PATH" \
    "$AP_RANGE_FILE" "$NF_RANGE_FILE"; do
    require_file "$file"
done
mkdir -p "$PREPROCESSED_DIR" "$PRETRAINING_OUTPUT_DIR"

echo "Job ${SLURM_JOB_ID:-interactive}; loss=$LOSS_TYPE; masking=$MASKING_STRATEGY; tag=$RUN_TAG"
if [[ "$REBUILD_DATA" == 1 ]]; then
    echo 'Phase 1: GC extraction, AP/NF/GC preprocessing and frequency files'
    python src/data_preprocessing/extract_gc_cohort.py --data-root "$RAW_DATA_DIR"
    python src/data_preprocessing/preprocess_chemo.py \
        --data-root "$RAW_DATA_DIR" --output-dir "$PREPROCESSED_DIR" --cohorts AP NF GC
    for cohort in AP NF; do
        for weight_type in inverse proportional; do
            python -m src.utils.compute_weights \
                --cohort "$cohort" --weight-type "$weight_type" \
                --preprocessed-dir "$PREPROCESSED_DIR" \
                --top-features-path "$TOP_FEATURES_PATH" \
                --output-dir "$PREPROCESSED_DIR"
        done
    done
else
    echo 'Phase 1: reusing existing processed fold files and frequency files'
fi
for cohort in AP NF; do
    for i in {0..4}; do
        require_file "$PREPROCESSED_DIR/chemo_${cohort}_fold_${i}.pkl"
    done
done
require_file "$GC_PREPROCESSED_PATH"

if [[ "$SKIP_FORMATTING" == 0 ]]; then
    echo 'Formatting GC tensors/masks using the specified configuration file'
    # The formatter reads this YAML, not Hydra CLI overrides. Check YAML masking strategy.
    python - "$PT_CONFIG" "$MASKING_STRATEGY" "$PT_DATA" "$PT_MASK" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.load(sys.argv[1])
expected = sys.argv[2]
if str(cfg.masking_strategy) != expected:
    raise SystemExit(f"YAML masking_strategy={cfg.masking_strategy!r}; requested={expected!r}. Update YAML or use the matching MASKING_STRATEGY.")
from pathlib import Path
root = Path.cwd()
for key, actual in (("data_path", sys.argv[3]), ("mask_path", sys.argv[4])):
    configured = Path(str(cfg[key]))
    configured = configured if configured.is_absolute() else root / configured
    if configured.resolve() != Path(actual).resolve():
        raise SystemExit(f"YAML {key}={configured} does not match formatter output {actual}")
PY
    python -m src.pretraining_data_preparation.format_and_mask_chemo \
        --config "$PT_CONFIG" --preprocessed-path "$GC_PREPROCESSED_PATH" \
        --top-features-path "$TOP_FEATURES_PATH" \
        --range-files "$AP_RANGE_FILE" "$NF_RANGE_FILE" \
        --output-dir "$PRETRAINING_OUTPUT_DIR"
else
    echo 'Skipping formatter: verify that existing GC mask matches this strategy.'
fi
require_file "$PT_DATA"
require_file "$PT_MASK"

# Standard/temporal/clinical: one GC checkpoint serves both cohorts.
# Frequency-weighted: pretrain GC separately with AP-derived and NF-derived weights.
pretrain_for() {
    local cohort="$1"
    local base="$ROOT/pretrained_models/Chemo/EMIT_MAE/GC/$RUN_TAG/${cohort}/EMIT_Chemo_GC_128d"
    local -a extra=()
    if [[ "$LOSS_TYPE" == inverse_frequency ]]; then
        local weights="$PREPROCESSED_DIR/inverse_freq_weights_${cohort}.pkl"
        require_file "$weights"
        extra+=("inverse_frequency_weights_path=$weights")
    elif [[ "$LOSS_TYPE" == proportional_frequency ]]; then
        local weights="$PREPROCESSED_DIR/proportional_freq_weights_${cohort}.pkl"
        require_file "$weights"
        extra+=("proportional_frequency_weights_path=$weights")
    fi
    echo "GC pretraining: cohort-weight source=$cohort; checkpoint=$base"
    python -m src.pretraining.pretrain_chemo --config-name pt_config_chemo_GC \
        "loss_type=$LOSS_TYPE" "model_name=$base" "${extra[@]}"
    require_file "${base}.weights.h5"
}

finetune_for() {
    local cohort="$1"
    local checkpoint="$2"
    require_file "$checkpoint"
    for i in {0..4}; do
        local result="$ROOT/results/Chemo/EMIT_MAE/testing/${cohort}/${RUN_TAG}/FINAL_Chemo_${cohort}_Fold_${i}.pkl"
        local fold="$PREPROCESSED_DIR/chemo_${cohort}_fold_${i}.pkl"
        echo "Fine-tuning $cohort fold $i from $checkpoint"
        python -m src.finetuning.finetune_chemo \
            --config-name "ft_config_chemo_${cohort}" \
            "data_path=$fold" \
            "pt_model_weights_path=$checkpoint" \
            "use_pretrained_weights=true" \
            "results_path=$result" \
            "file_name=FINAL_Chemo_${cohort}_Fold_${i}"
        require_file "$result"
    done
}

if [[ "$LOSS_TYPE" == inverse_frequency || "$LOSS_TYPE" == proportional_frequency ]]; then
    for cohort in AP NF; do
        pretrain_for "$cohort"
        checkpoint="$ROOT/pretrained_models/Chemo/EMIT_MAE/GC/$RUN_TAG/${cohort}/EMIT_Chemo_GC_128d.weights.h5"
        finetune_for "$cohort" "$checkpoint"
    done
else
    pretrain_for shared
    checkpoint="$ROOT/pretrained_models/Chemo/EMIT_MAE/GC/$RUN_TAG/shared/EMIT_Chemo_GC_128d.weights.h5"
    finetune_for AP "$checkpoint"
    finetune_for NF "$checkpoint"
fi

echo "Completed $RUN_TAG"
