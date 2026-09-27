"""
Compute frequency-derived reconstruction weights.

This script calculates variable occurrence counts from the full preprocessed
Aplasia (AP) or Neutropenic Fever (NF) cohort package and converts them into
inverse-frequency or proportional-frequency reconstruction weights.

The output is used by the frequency-weighted reconstruction-loss experiments
reported in the thesis.

Important limitation
--------------------
Counts are computed from the full AP or NF event table, including event records
from admissions assigned to train, validation, and test partitions. Outcome
labels are not used. However, this procedure may expose pretraining to
feature-distribution information from validation and test admissions. It is
retained here to reproduce the reported thesis experiments.

For a stricter leakage-controlled extension, calculate counts from the external
General Cancer (GC) pretraining cohort or separately from each active
fine-tuning fold's training admissions.

All input packages and output weight files are derived from restricted MIMIC-IV data and cannot be committed to a public repository.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd


TOP_FEATURE_COUNT = 100
VALID_COHORTS = ("AP", "NF")
VALID_WEIGHT_TYPES = ("inverse", "proportional")


def parse_args() -> argparse.Namespace:
    """Parse paths, downstream cohort, and frequency-weighting strategy."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute full-cohort frequency weights for AP/NF experiments."
        )
    )
    parser.add_argument(
        "--cohort",
        choices=VALID_COHORTS,
        required=True,
        help="Downstream cohort whose full event table is counted.",
    )
    parser.add_argument(
        "--weight-type",
        choices=VALID_WEIGHT_TYPES,
        default="inverse",
        help="Frequency weighting strategy. Default: inverse.",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        required=True,
        help="Restricted directory containing `chemo_AP_fold_0.pkl` or `chemo_NF_fold_0.pkl`.",
    )
    parser.add_argument(
        "--top-features-path",
        type=Path,
        required=True,
        help="Restricted path to `mimic_top100_features.pkl`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Restricted directory where the generated weight pickle is saved.",
    )
    return parser.parse_args()


def validate_paths(paths: list[Path]) -> None:
    """Raise a clear error when required restricted input files are unavailable."""
    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        formatted = "\n".join(f" - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing required input files:\n{formatted}")


def build_master_vocabulary(top_features_path: Path) -> dict[str, int]:
    """
    Build the fixed 102-variable mapping used by the final GC pretraining pipeline.

    The mapping consists of the supplied top-100 clinical variable identifiers
    plus the static Age and Gender event variables. Integer index 0 is reserved for padding.
    """
    with top_features_path.open("rb") as handle:
        top_features = pickle.load(handle)

    vocabulary = sorted([str(item) for item in top_features] + ["Age", "Gender"])

    if len(vocabulary) != TOP_FEATURE_COUNT + 2:
        raise ValueError(
            f"Expected {TOP_FEATURE_COUNT + 2} variables, found {len(vocabulary)}."
        )

    return {
        variable: index + 1
        for index, variable in enumerate(vocabulary)
    }


def load_full_event_table(package_path: Path) -> pd.DataFrame:
    """
    Load the full event table from an AP/NF fold-0 preprocessing package.

    The first element of the saved package is the complete cohort event table,
    not only the fold-0 training subset.
    """
    with package_path.open("rb") as handle:
        events, _, _, _, _ = pickle.load(handle)

    required_columns = {"variable", "ts_ind", "value", "hour"}
    missing_columns = required_columns - set(events.columns)
    if missing_columns:
        raise ValueError(
            f"Event table is missing required columns: {sorted(missing_columns)}"
        )

    events = events.copy()
    events["variable"] = events["variable"].astype(str)

    if events.empty:
        raise RuntimeError("The input event table is empty.")

    return events


def calculate_weights(
    events: pd.DataFrame,
    variable_to_index: dict[str, int],
    weight_type: str,
) -> dict[int, float]:
    """
    Calculate fixed-vocabulary inverse or proportional frequency weights.

    Counts represent total retained event records per variable across all
    admissions in the supplied AP/NF cohort package. They are not admission
    counts and do not use outcome labels.
    """
    counts = events["variable"].value_counts()

    unexpected_variables = set(counts.index) - set(variable_to_index)
    if unexpected_variables:
        raise ValueError(
            "Event table contains variables absent from the fixed vocabulary: "
            f"{sorted(unexpected_variables)[:10]}"
        )

    weight_series = pd.Series(
        0.0,
        index=list(variable_to_index.keys()),
        dtype=float,
    )
    observed_counts = counts.reindex(weight_series.index, fill_value=0).astype(float)

    positive_counts = observed_counts > 0
    if not positive_counts.any():
        raise RuntimeError("No observed variables have positive event counts.")

    if weight_type == "inverse":
        total_count = observed_counts[positive_counts].sum()
        normalized_frequency = observed_counts[positive_counts] / total_count
        weight_series.loc[positive_counts] = 1.0 / normalized_frequency

    elif weight_type == "proportional":
        weight_series.loc[positive_counts] = observed_counts[positive_counts]

    else:
        raise ValueError(
            f"Unsupported weight type '{weight_type}'. "
            f"Choose one of: {', '.join(VALID_WEIGHT_TYPES)}."
        )

    weight_sum = weight_series.sum()
    if weight_sum <= 0:
        raise RuntimeError("Computed frequency weights have non-positive total.")

    weight_series /= weight_sum

    return {
        variable_to_index[variable]: float(weight)
        for variable, weight in weight_series.items()
        if weight > 0
    }


def main() -> None:
    """Compute and save one AP/NF frequency-weight dictionary."""
    args = parse_args()

    preprocessed_dir = args.preprocessed_dir.expanduser().resolve()
    top_features_path = args.top_features_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    package_path = preprocessed_dir / f"chemo_{args.cohort}_fold_0.pkl"
    output_path = output_dir / f"{args.weight_type}_freq_weights_{args.cohort}.pkl"

    validate_paths([package_path, top_features_path])

    print(
        f"Computing {args.weight_type}-frequency weights from the full "
        f"{args.cohort} cohort event table."
    )
    print(
        "Note: this reproduces the frequency-statistic procedure and is not fold-isolated."
    )

    variable_to_index = build_master_vocabulary(top_features_path)
    events = load_full_event_table(package_path)
    weights = calculate_weights(
        events=events,
        variable_to_index=variable_to_index,
        weight_type=args.weight_type,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(weights, handle)

    print(f"Saved {args.weight_type}-frequency weights: {output_path}")
    print(f"Variables with non-zero weights: {len(weights)}")


if __name__ == "__main__":
    main()