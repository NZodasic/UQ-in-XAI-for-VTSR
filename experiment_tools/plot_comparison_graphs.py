import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _numeric(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _label_series(df):
    labels = df["model_name"].fillna("model").astype(str)
    if "experiment_group" in df.columns:
        labels = df["experiment_group"].fillna("run").astype(str) + ": " + labels
    if "uq_label" in df.columns:
        labels = labels + " | " + df["uq_label"].fillna("").astype(str)
    if "xai_label" in df.columns:
        labels = labels + " | " + df["xai_label"].fillna("").astype(str)
    return labels


def _bar_chart(df, x_column, y_columns, title, ylabel, output_path):
    available = [column for column in y_columns if column in df.columns and df[column].notna().any()]
    if not available or df.empty:
        return False

    plot_df = df[[x_column] + available].dropna(how="all", subset=available).copy()
    if plot_df.empty:
        return False

    ax = plot_df.plot(
        x=x_column,
        y=available,
        kind="bar",
        figsize=(max(10, len(plot_df) * 0.8), 5),
        rot=35
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return True


def plot_graphs(input_csv, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    numeric_columns = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "ece",
        "mce",
        "nll",
        "brier",
        "latency_ms",
        "xai_time_ms",
        "xai_fidelity",
        "xai_stability",
        "num_params",
        "num_params_m",
        "size_mb",
        "best_val_accuracy",
    ]
    df = _numeric(df, numeric_columns)
    df["plot_label"] = _label_series(df)

    written = []

    groups = df["experiment_group"].dropna().unique().tolist() if "experiment_group" in df.columns else []

    if "backbone" in groups:
        backbone = df[df["experiment_group"] == "backbone"].copy()
    else:
        backbone = df.copy()
    if _bar_chart(
        backbone,
        "plot_label",
        ["accuracy", "f1_macro"],
        "Backbone Accuracy and F1-macro",
        "Score",
        output_dir / "backbone_accuracy_f1.png",
    ):
        written.append(output_dir / "backbone_accuracy_f1.png")
    if _bar_chart(
        backbone,
        "plot_label",
        ["num_params_m", "latency_ms"],
        "Backbone Parameters and Inference Time",
        "Params (M) / ms",
        output_dir / "backbone_params_latency.png",
    ):
        written.append(output_dir / "backbone_params_latency.png")

    uq = df[df["experiment_group"] == "uq"].copy() if "uq" in groups else df.copy()
    if _bar_chart(
        uq,
        "plot_label",
        ["ece", "mce", "nll", "brier"],
        "UQ and Calibration Metrics",
        "Lower is better",
        output_dir / "uq_calibration_metrics.png",
    ):
        written.append(output_dir / "uq_calibration_metrics.png")
    if _bar_chart(
        uq,
        "plot_label",
        ["accuracy", "f1_macro", "latency_ms"],
        "UQ Accuracy, F1, and Runtime",
        "Score / ms",
        output_dir / "uq_accuracy_runtime.png",
    ):
        written.append(output_dir / "uq_accuracy_runtime.png")

    xai = df[df["experiment_group"] == "xai"].copy() if "xai" in groups else df.copy()
    if _bar_chart(
        xai,
        "plot_label",
        ["xai_fidelity", "xai_stability"],
        "XAI Fidelity and Stability",
        "Score",
        output_dir / "xai_fidelity_stability.png",
    ):
        written.append(output_dir / "xai_fidelity_stability.png")
    if _bar_chart(
        xai,
        "plot_label",
        ["xai_time_ms"],
        "XAI Computation Time",
        "ms/sample",
        output_dir / "xai_time.png",
    ):
        written.append(output_dir / "xai_time.png")

    return written


def main():
    parser = argparse.ArgumentParser(description="Plot comparison graphs from benchmark_results.csv.")
    parser.add_argument("--input", default="EXPERIMENT/benchmark_results.csv")
    parser.add_argument("--output-dir", default="EXPERIMENT/graphs")
    args = parser.parse_args()

    written = plot_graphs(args.input, args.output_dir)
    if written:
        for path in written:
            print(f"Wrote {path}")
    else:
        print("No graphs written. Check that the input CSV has completed metric rows.")


if __name__ == "__main__":
    main()
