import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _run_command(command):
    print("Running:", " ".join(command), flush=True)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as exc:
        if any("data.num_workers=" in part and not part.endswith("=0") for part in command):
            print(
                "Command failed. On Windows, DataLoader workers can exhaust shared "
                "file mappings; retry with --num-workers 0.",
                flush=True
            )
        raise


def _latest_run_dir(experiment_dir):
    runs = sorted(Path(experiment_dir).glob("run_*"))
    if not runs:
        raise RuntimeError(f"No run directories found in {experiment_dir}")
    return runs[-1]


def _append_metadata(summary_csv, metadata):
    summary_csv = Path(summary_csv)
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    for key in metadata:
        if key not in fieldnames:
            fieldnames.append(key)

    for row in rows:
        row.update(metadata)

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _collect_run_summaries(run_dirs, output_path):
    rows = []
    fieldnames = []

    for run_dir in run_dirs:
        summary_csv = Path(run_dir) / "metrics_summary.csv"
        with summary_csv.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                for field in reader.fieldnames or []:
                    if field not in fieldnames:
                        fieldnames.append(field)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), output_path


def _safe_name(value):
    value = str(value or "").strip().lower()
    value = value.replace("+", "plus")
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("._-") or "run"


def _run_number_prefix(run_dir):
    match = re.match(r"^(run_\d+)", Path(run_dir).name)
    return match.group(1) if match else Path(run_dir).name


def _benchmark_metadata(item, index, total, run_dir):
    group = item.get("group", "")
    label = item.get("label", "")
    model_name = item.get("model", "")
    if group == "backbone":
        method_label = _model_label(model_name)
    elif group == "uq":
        method_label = _method_label({"benchmark_label": label})
    elif group == "xai":
        method_label = _xai_label({"benchmark_label": label})
    else:
        method_label = label

    return {
        "run_dir": str(run_dir),
        "benchmark_run_name": Path(run_dir).name,
        "benchmark_index": index,
        "benchmark_total": total,
        "experiment_group": group,
        "benchmark_label": label,
        "benchmark_model": model_name,
        "model_label": _model_label(model_name),
        "method_label": method_label,
        "benchmark_display_name": " / ".join(
            part for part in [group, _model_label(model_name), method_label] if part
        ),
        "uq_label": label if group == "uq" else "",
        "xai_label": label if group == "xai" else "",
    }


def _rename_run_dir(run_dir, item):
    run_dir = Path(run_dir)
    model_name = item.get("model", "")
    parts = [
        _run_number_prefix(run_dir),
        _safe_name(item.get("group", "")),
        _safe_name(model_name),
        _safe_name(item.get("label", "")),
    ]
    target = run_dir.with_name("_".join(part for part in parts if part))
    if target == run_dir:
        return run_dir

    candidate = target
    suffix = 2
    while candidate.exists():
        candidate = target.with_name(f"{target.name}_{suffix}")
        suffix += 1

    try:
        run_dir.rename(candidate)
        print(f"Renamed run folder: {run_dir.name} -> {candidate.name}", flush=True)
        return candidate
    except OSError as exc:
        print(f"Warning: could not rename {run_dir} to {candidate}: {exc}", flush=True)
        return run_dir


def _item_from_summary(row):
    group = row.get("experiment_group") or "manual"
    label = row.get("benchmark_label") or row.get("xai_variant") or row.get("uq_method") or row.get("model_name") or "run"
    return {
        "group": group,
        "label": label,
        "model": row.get("benchmark_model") or row.get("model_name", ""),
    }


def collect_existing_runs(experiment_dir, output_path, rename_runs=True):
    run_dirs = []
    candidates = sorted(Path(experiment_dir).glob("run_*"))
    total = len([path for path in candidates if (path / "metrics_summary.csv").exists()])

    for index, run_dir in enumerate(candidates, start=1):
        summary_csv = run_dir / "metrics_summary.csv"
        if not summary_csv.exists():
            continue

        with summary_csv.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        item = _item_from_summary(rows[0])
        named_run_dir = _rename_run_dir(run_dir, item) if rename_runs else run_dir
        metadata = _benchmark_metadata(item, index, total, named_run_dir)
        _append_metadata(named_run_dir / "metrics_summary.csv", metadata)
        run_dirs.append(named_run_dir)

    return _collect_run_summaries(run_dirs, output_path)


def _to_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_float(value, digits=4):
    value = _to_float(value)
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _format_params(value):
    value = _to_float(value)
    if value is None:
        return ""
    return f"{value:.1f}"


def _write_table(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _load_rows(csv_path):
    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _method_label(row):
    label = row.get("benchmark_label", "")
    labels = {
        "no_calibration": "No Calibration",
        "label_smoothing": "Label Smoothing (eps=0.1)",
        "temperature_scaling": "Temperature Scaling",
        "mc_dropout": "MC Dropout (T=20)",
        "mc_dropout_temperature_scaling": "TS + MC Dropout",
    }
    return labels.get(label, label)


def _model_label(model_name):
    labels = {
        "mobilenet_v2": "MobileNetV2",
        "resnet50": "ResNet-50",
        "densenet121": "DenseNet-121",
        "efficientnet_b2": "EfficientNet-B2",
    }
    return labels.get(model_name, model_name)


def _xai_label(row):
    label = row.get("benchmark_label", "")
    labels = {
        "gradcam": "GradCAM",
        "gradcam++": "GradCAM++",
        "eigencam": "EigenCAM",
        "hirescam": "HiResCAM",
        "integrated_gradients": "Integrated Gradients",
    }
    return labels.get(label, label)


def export_thesis_tables(benchmark_csv, output_dir):
    rows = _load_rows(benchmark_csv)
    output_dir = Path(output_dir)

    uq_rows = [
        row for row in rows
        if row.get("experiment_group") == "uq"
    ]
    uq_table = []
    for row in uq_rows:
        uq_table.append({
            "Method": _method_label(row),
            "Accuracy ↑": _format_float(row.get("accuracy")),
            "ECE ↓": _format_float(row.get("ece")),
            "MCE ↓": _format_float(row.get("mce")),
            "NLL ↓": _format_float(row.get("nll")),
            "Brier ↓": _format_float(row.get("brier")),
            "Infer. Time": _format_float(row.get("latency_ms"), digits=2),
        })

    xai_rows = [
        row for row in rows
        if row.get("experiment_group") == "xai"
    ]
    xai_table = []
    for row in xai_rows:
        xai_table.append({
            "Method": _xai_label(row),
            "Fidelity ↑": _format_float(row.get("xai_fidelity")),
            "Stability ↑": _format_float(row.get("xai_stability")),
            "Comp. Time (ms)": _format_float(row.get("xai_time_ms"), digits=2),
        })

    backbone_rows = [
        row for row in rows
        if row.get("experiment_group") == "backbone"
    ]
    model_table = []
    for row in backbone_rows:
        model_table.append({
            "Model": _model_label(row.get("model_name", "")),
            "Accuracy ↑": _format_float(row.get("accuracy")),
            "F1-macro ↑": _format_float(row.get("f1_macro")),
            "Params (M)": _format_params(row.get("num_params_m")),
            "Infer. Time (ms)": _format_float(row.get("latency_ms"), digits=2),
            "ECE ↓": _format_float(row.get("ece")),
            "NLL ↓": _format_float(row.get("nll")),
        })

    outputs = [
        _write_table(
            output_dir / "Method-Accuracy-ECE-MCE-NLL-Brier-InferTime.csv",
            ["Method", "Accuracy ↑", "ECE ↓", "MCE ↓", "NLL ↓", "Brier ↓", "Infer. Time"],
            uq_table,
        ),
        _write_table(
            output_dir / "Method-Fidelity-Stability-CompTimems.csv",
            ["Method", "Fidelity ↑", "Stability ↑", "Comp. Time (ms)"],
            xai_table,
        ),
        _write_table(
            output_dir / "Model-Accuracy-F1-macro-ParamsM-InferTimems-ECE-NL.csv",
            ["Model", "Accuracy ↑", "F1-macro ↑", "Params (M)", "Infer. Time (ms)", "ECE ↓", "NLL ↓"],
            model_table,
        ),
    ]
    return outputs


def _main_command(config, overrides):
    command = [sys.executable, "-u", "main.py", "--config", config]
    for override in overrides:
        command.extend(["--set", override])
    return command


def _run_suffix(item):
    return "_".join(
        part for part in [
            _safe_name(item.get("group", "")),
            _safe_name(item.get("model", "")),
            _safe_name(item.get("label", "")),
        ] if part
    )


def benchmark_plan(args):
    quick_overrides = []
    if args.epochs is not None:
        quick_overrides.append(f"training.epochs={args.epochs}")
        quick_overrides.append(f"training.patience={args.patience}")
    if args.batch_size is not None:
        quick_overrides.append(f"data.batch_size={args.batch_size}")
    if args.num_workers is not None:
        quick_overrides.append(f"data.num_workers={args.num_workers}")
    if args.prefetch_factor is not None:
        quick_overrides.append(f"data.prefetch_factor={args.prefetch_factor}")
    if args.uncertainty_samples is not None:
        quick_overrides.append(f"uncertainty.num_samples={args.uncertainty_samples}")
    if args.xai_mc_samples is not None:
        quick_overrides.append(f"explainability.mc_samples={args.xai_mc_samples}")
    if args.ig_steps is not None:
        quick_overrides.append(f"explainability.steps={args.ig_steps}")

    common = [
        f"experiment.device={args.device}",
        "experiment.save_config=true",
        "experiment.save_summary=true",
        "training.log_epoch_metrics=true",
        f"training.checkpoint.save_every_epochs={args.checkpoint_every}",
        f"training.tensorboard.enabled={str(args.tensorboard).lower()}",
        f"explainability.compute_metrics={str(not args.skip_xai_metrics).lower()}",
    ] + quick_overrides

    plan = []

    if args.suite in {"full", "backbone"}:
        for model_name in ["mobilenet_v2", "resnet50", "densenet121", "efficientnet_b2"]:
            plan.append({
                "group": "backbone",
                "label": model_name,
                "model": model_name,
                "overrides": common + [
                    f"model.name={model_name}",
                    "uncertainty.method=mc_dropout",
                    "calibration.temperature_scaling=true",
                    "explainability.generate=false",
                    "explainability.method=gradcam",
                    "explainability.variant=gradcam++",
                ],
            })

    if args.suite in {"full", "uq"}:
        uq_runs = [
            ("no_calibration", [
                "uncertainty.method=none",
                "calibration.temperature_scaling=false",
                "training.label_smoothing=0.0",
            ]),
            ("label_smoothing", [
                "uncertainty.method=none",
                "calibration.temperature_scaling=false",
                "training.label_smoothing=0.1",
            ]),
            ("temperature_scaling", [
                "uncertainty.method=none",
                "calibration.temperature_scaling=true",
                "training.label_smoothing=0.0",
            ]),
            ("mc_dropout", [
                "uncertainty.method=mc_dropout",
                "calibration.temperature_scaling=false",
                "training.label_smoothing=0.1",
            ]),
            ("mc_dropout_temperature_scaling", [
                "uncertainty.method=mc_dropout",
                "calibration.temperature_scaling=true",
                "training.label_smoothing=0.1",
            ]),
        ]
        for label, overrides in uq_runs:
            plan.append({
                "group": "uq",
                "label": label,
                "model": args.uq_model,
                "overrides": common + [
                    f"model.name={args.uq_model}",
                    "explainability.generate=false",
                    "explainability.method=gradcam",
                    "explainability.variant=gradcam++",
                ] + overrides,
            })

    if args.suite in {"full", "xai"}:
        xai_runs = [
            ("gradcam", ["explainability.method=gradcam", "explainability.variant=gradcam"]),
            ("gradcam++", ["explainability.method=gradcam", "explainability.variant=gradcam++"]),
            ("eigencam", ["explainability.method=gradcam", "explainability.variant=eigencam"]),
            ("hirescam", ["explainability.method=gradcam", "explainability.variant=hirescam"]),
            ("integrated_gradients", ["explainability.method=integrated_gradients"]),
        ]
        for label, overrides in xai_runs:
            plan.append({
                "group": "xai",
                "label": label,
                "model": args.xai_model,
                "overrides": common + [
                    f"model.name={args.xai_model}",
                    "uncertainty.method=mc_dropout",
                    "calibration.temperature_scaling=true",
                    "explainability.generate=true",
                ] + overrides,
            })

    if args.only_labels:
        wanted = {
            label.strip()
            for raw_label in args.only_labels
            for label in raw_label.split(",")
            if label.strip()
        }
        plan = [item for item in plan if item["label"] in wanted]

    return plan


def _format_run_budget(args):
    budget = []
    if args.epochs is not None:
        budget.append(f"epochs={args.epochs}")
        budget.append(f"patience={args.patience}")
    if args.batch_size is not None:
        budget.append(f"batch_size={args.batch_size}")
    if args.num_workers is not None:
        budget.append(f"num_workers={args.num_workers}")
    if args.prefetch_factor is not None:
        budget.append(f"prefetch_factor={args.prefetch_factor}")
    if args.uncertainty_samples is not None:
        budget.append(f"uq_samples={args.uncertainty_samples}")
    if args.xai_mc_samples is not None:
        budget.append(f"xai_mc_samples={args.xai_mc_samples}")
    if args.ig_steps is not None:
        budget.append(f"ig_steps={args.ig_steps}")
    if args.skip_xai_metrics:
        budget.append("xai_metrics=off")
    return ", ".join(budget) if budget else "config defaults"


def main():
    parser = argparse.ArgumentParser(
        description="Run supported backbone, UQ, and XAI comparison experiments in one command."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--experiment-dir", default="EXPERIMENT")
    parser.add_argument("--suite", choices=["full", "backbone", "uq", "xai"], default="full")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=None, help="Optional quick-run epoch override.")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=None)
    parser.add_argument("--uncertainty-samples", type=int, default=None)
    parser.add_argument("--xai-mc-samples", type=int, default=None)
    parser.add_argument("--ig-steps", type=int, default=None)
    parser.add_argument("--uq-model", default="efficientnet_b2")
    parser.add_argument("--xai-model", default="efficientnet_b2")
    parser.add_argument(
        "--only-labels",
        action="append",
        default=[],
        help="Run only matching benchmark labels, comma-separated or repeated. Example: --only-labels eigencam,hirescam",
    )
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--skip-xai-metrics", action="store_true")
    parser.add_argument("--collect-existing", action="store_true")
    parser.add_argument("--no-rename-runs", action="store_true")
    parser.add_argument("--export-thesis-tables", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    comparison_csv = Path(args.experiment_dir) / "benchmark_results.csv"
    if args.collect_existing:
        rows, output_path = collect_existing_runs(
            args.experiment_dir,
            comparison_csv,
            rename_runs=not args.no_rename_runs
        )
        print(f"Collected {rows} rows into {output_path}")
        return

    plan = benchmark_plan(args)
    if not plan:
        raise RuntimeError("Benchmark plan is empty.")
    print(f"Benchmark budget: {_format_run_budget(args)}", flush=True)

    run_dirs = []
    completed_count = 0
    for index, item in enumerate(plan, start=1):
        print(f"[{index}/{len(plan)}] {item['group']} - {item['label']}", flush=True)
        command = _main_command(args.config, item["overrides"])
        command.extend(["--run-suffix", _run_suffix(item)])
        if args.dry_run:
            print(" ".join(command))
            continue

        _run_command(command)
        run_dir = _latest_run_dir(args.experiment_dir)
        run_dir = _rename_run_dir(run_dir, item)
        run_dirs.append(run_dir)
        summary_csv = run_dir / "metrics_summary.csv"
        _append_metadata(
            summary_csv,
            _benchmark_metadata(item, index, len(plan), run_dir)
        )
        completed_count, output_path = _collect_run_summaries(run_dirs, comparison_csv)
        print(f"Updated combined CSV: {output_path} ({completed_count} rows)", flush=True)

    if args.dry_run:
        return

    rows, output_path = _collect_run_summaries(run_dirs, comparison_csv)
    print(f"Collected {rows} rows into {output_path}")

    if args.export_thesis_tables:
        table_paths = export_thesis_tables(output_path, args.output_dir)
        for path in table_paths:
            print(f"Wrote {path}")

    from experiment_tools.plot_comparison_graphs import plot_graphs

    graph_dir = Path(args.experiment_dir) / "graphs"
    written = plot_graphs(output_path, graph_dir)
    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
