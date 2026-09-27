# Code accompanying the Master's thesis:

> Advancing representation learning of clinical time series via multi-aware transformers

The project evaluates continuous-time masked autoencoding and downstream clinical event prediction on sparse oncology time-series data.

## Repository contents

- `src/model.py`: Model architecture.
- `src/data_preprocessing/`: cohort extraction and preprocessing.
- `src/pretraining_data_preparation/`: tensor formatting and event masking.
- `src/pretraining/`: GC self-supervised pretraining.
- `src/finetuning/`: AP/NF downstream fine-tuning.
- `src/utils/`: frequency-weight generation and five-fold AP/NF metric aggregation.
- `config/`: GC pretraining and AP/NF fine-tuning configurations.
- `run_master_pipeline.sh`: SLURM orchestration script.

## Configuration and outputs

Set the GC pretraining loss and masking strategy in
`config/pt_config_chemo_GC.yaml`. Set AP and NF observation windows,
sequence lengths, batching, and optimization in
`config/ft_config_chemo_AP.yaml` and `config/ft_config_chemo_NF.yaml`.
A single submission runs the selected configuration; it does not sweep
all configurations reported in the thesis.

By default, the master script rebuilds cohort packages and GC masks.
For inverse- or proportional-frequency reconstruction loss, it runs
separate GC pretraining passes with AP- and NF-derived frequency
statistics, respectively, then fine-tunes the matching downstream
cohort. Other loss types use one shared GC checkpoint.

The script creates run-tagged checkpoint and result directories. It
overrides the `model_name` and fold-specific `data_path`,
`pt_model_weights_path`, and `results_path` defaults in the YAML files.
Output artifacts are restricted and must not be published.

## Data access

This repository does not contain MIMIC-IV, AP, NF, or GC data; generated tensors, masks, checkpoints, and result files are also excluded.

MIMIC-IV v3.1 is available through PhysioNet under credentialed access:
https://physionet.org/content/mimiciv/3.1/ 

Researchers must obtain the required PhysioNet access, complete the applicable
training, and comply with the data-use agreement. AP and NF input packages and
fold definitions must be obtained through the source protocol described in the
thesis.

## Workflow

The final workflow is:

GC cohort pretraining -> AP fine-tuning
GC cohort pretraining -> NF fine-tuning

The GC configuration controls the pretraining objective, masking strategy,
architecture, optimization, and restricted local artifact paths. The AP and NF
configurations control downstream observation windows, sequence lengths,
batching, optimization, and result paths.

## Running on SLURM

Edit the restricted local paths and configurations as necessary, then submit:

```bash
sbatch run_master_pipeline.sh
```

## Reproducibility

Record the Git tag, YAML files, seed, TensorFlow/CUDA/cuDNN versions, GPU model,
and data-access package versions for each experiment. GPU-level deterministic
settings reduce but do not guarantee bitwise-identical results across software
and hardware environments.
