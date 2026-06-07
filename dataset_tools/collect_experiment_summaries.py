import argparse
import csv
from pathlib import Path


def collect_summaries(experiment_dir, output_path):
    summary_paths = sorted(Path(experiment_dir).glob("run_*/metrics_summary.csv"))
    rows = []
    fieldnames = []

    for summary_path in summary_paths:
        with summary_path.open("r", newline="", encoding="utf-8") as f:
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


def main():
    parser = argparse.ArgumentParser(
        description="Collect per-run metrics_summary.csv files into one comparison table."
    )
    parser.add_argument("--experiment-dir", default="EXPERIMENT")
    parser.add_argument("--output", default="EXPERIMENT/comparison_table.csv")
    args = parser.parse_args()

    num_rows, output_path = collect_summaries(args.experiment_dir, args.output)
    print(f"Wrote {num_rows} rows to {output_path}")


if __name__ == "__main__":
    main()
