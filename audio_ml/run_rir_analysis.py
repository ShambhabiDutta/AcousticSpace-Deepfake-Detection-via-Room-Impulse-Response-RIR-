from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon


ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / "features"
INPUT = FEATURES / "sample_features_with_rir_metadata.csv"

CONDITIONS = ["clean", "reverb", "noisy_snr10", "noisy_snr5", "noisy_snr0"]
METRICS = {
    "rms": "RMS",
    "zcr": "Zero-crossing rate",
    "spectral_centroid_hz": "Spectral centroid (Hz)",
}


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values, returned in original order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p), dtype=float)
    running_max = 0.0
    n = len(p)

    for rank, idx in enumerate(order):
        value = min(1.0, (n - rank) * p[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max

    return adjusted


def main():
    FEATURES.mkdir(parents=True, exist_ok=True)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Required input not found: {INPUT}\n"
            "Create sample_features_with_rir_metadata.csv first."
        )

    df = pd.read_csv(INPUT)

    required = {
        "sample_id", "condition", "rms", "zcr",
        "spectral_centroid_hz", "t60_s",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")

    df = df[df["condition"].isin(CONDITIONS)].copy()
    if df.empty:
        raise ValueError("No recognized acoustic conditions were found.")

    # 1. Summary statistics by acoustic condition.
    summary = (
        df.groupby("condition", observed=True)
        .agg(
            n=("sample_id", "count"),
            rms_mean=("rms", "mean"),
            rms_std=("rms", "std"),
            zcr_mean=("zcr", "mean"),
            zcr_std=("zcr", "std"),
            spectral_centroid_mean_hz=("spectral_centroid_hz", "mean"),
            spectral_centroid_std_hz=("spectral_centroid_hz", "std"),
        )
        .reindex(CONDITIONS)
        .dropna(how="all")
        .reset_index()
    )
    summary.to_csv(FEATURES / "condition_feature_summary.csv", index=False)

    # 2. Compare feature distributions across conditions.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (column, label) in zip(axes, METRICS.items()):
        data = [
            df.loc[df["condition"] == condition, column].dropna().to_numpy()
            for condition in CONDITIONS
        ]
        labels = [c for c, values in zip(CONDITIONS, data) if len(values)]
        data = [values for values in data if len(values)]
        ax.boxplot(data, tick_labels=labels)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Acoustic Features by Condition")
    fig.tight_layout()
    fig.savefig(FEATURES / "sample_feature_comparison.png", dpi=150)
    fig.savefig(FEATURES / "feature_comparison_100_samples.png", dpi=150)
    plt.close(fig)

    # 3. Paired comparison: reverb minus clean for matching sample IDs.
    paired = df[df["condition"].isin(["clean", "reverb"])].pivot(
        index="sample_id",
        columns="condition",
        values=list(METRICS.keys()),
    )

    if not {"clean", "reverb"}.issubset(set(paired.columns.get_level_values(1))):
        raise ValueError("Both clean and reverb conditions are required.")

    changes = pd.DataFrame(index=paired.index)
    changes["rms_change"] = paired[("rms", "reverb")] - paired[("rms", "clean")]
    changes["zcr_change"] = paired[("zcr", "reverb")] - paired[("zcr", "clean")]
    changes["spectral_centroid_change_hz"] = (
        paired[("spectral_centroid_hz", "reverb")]
        - paired[("spectral_centroid_hz", "clean")]
    )
    changes.index.name = "sample_id"
    changes = changes.dropna().reset_index()
    changes.to_csv(FEATURES / "paired_reverb_changes.csv", index=False)

    # 4. Paired Wilcoxon tests with Holm correction.
    test_rows = []
    change_columns = [
        "rms_change",
        "zcr_change",
        "spectral_centroid_change_hz",
    ]

    for column in change_columns:
        values = changes[column].to_numpy(dtype=float)
        if len(values) == 0:
            raise ValueError(f"No paired values available for {column}.")

        if np.allclose(values, 0):
            statistic, p_value = 0.0, 1.0
        else:
            result = wilcoxon(values, alternative="two-sided")
            statistic, p_value = float(result.statistic), float(result.pvalue)

        test_rows.append({
            "feature": column,
            "n_pairs": len(values),
            "median_change": float(np.median(values)),
            "mean_change": float(np.mean(values)),
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
        })

    adjusted = holm_adjust([row["p_value"] for row in test_rows])
    for row, value in zip(test_rows, adjusted):
        row["holm_adjusted_p_value"] = float(value)

    pd.DataFrame(test_rows).to_csv(
        FEATURES / "paired_reverb_statistical_tests.csv", index=False
    )

    # 5. Plot paired changes.
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, column in zip(axes, change_columns):
        ax.boxplot(changes[column].dropna().to_numpy())
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(column.replace("_", " ").title())
        ax.set_xticks([])
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Paired Feature Changes: Reverb Minus Clean")
    fig.tight_layout()
    fig.savefig(FEATURES / "paired_reverb_changes.png", dpi=150)
    plt.close(fig)

    # 6. Correlate reverb-condition T60 with paired feature changes.
    reverb_meta = (
        df.loc[df["condition"] == "reverb", ["sample_id", "t60_s"]]
        .drop_duplicates("sample_id")
    )
    correlation_data = changes.merge(reverb_meta, on="sample_id", how="inner")
    correlation_rows = []

    for column in change_columns:
        valid = correlation_data[["t60_s", column]].dropna()
        if len(valid) < 3 or valid["t60_s"].nunique() < 2 or valid[column].nunique() < 2:
            rho, p_value = np.nan, np.nan
        else:
            result = spearmanr(valid["t60_s"], valid[column])
            rho, p_value = float(result.statistic), float(result.pvalue)

        correlation_rows.append({
            "feature_change": column,
            "n": len(valid),
            "spearman_rho": rho,
            "p_value": p_value,
        })

    pd.DataFrame(correlation_rows).to_csv(
        FEATURES / "t60_paired_change_correlations.csv", index=False
    )

    # 7. Plot T60 against paired changes.
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, column in zip(axes, change_columns):
        ax.scatter(correlation_data["t60_s"], correlation_data[column])
        ax.set_xlabel("T60 (seconds)")
        ax.set_ylabel(column.replace("_", " "))
        ax.grid(alpha=0.25)

    fig.suptitle("T60 vs Paired Acoustic Feature Changes")
    fig.tight_layout()
    fig.savefig(FEATURES / "t60_paired_change_scatter.png", dpi=150)
    plt.close(fig)

    # 8. Focused T60 versus spectral centroid plot for reverberant samples.
    reverb_rows = df[df["condition"] == "reverb"].dropna(
        subset=["t60_s", "spectral_centroid_hz"]
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(reverb_rows["t60_s"], reverb_rows["spectral_centroid_hz"])
    ax.set_xlabel("T60 (seconds)")
    ax.set_ylabel("Spectral centroid (Hz)")
    ax.set_title("T60 vs Spectral Centroid — Reverb Condition")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FEATURES / "t60_vs_spectral_centroid_reverb.png", dpi=150)
    plt.close(fig)

    print("RIR analysis completed successfully.")
    print(f"Input rows analyzed: {len(df)}")
    print(f"Paired samples analyzed: {len(changes)}")
    print(f"Outputs saved to: {FEATURES}")
    print("Note: this analyzes acoustic conditions; it does not evaluate deepfake detection.")


if __name__ == "__main__":
    main()
