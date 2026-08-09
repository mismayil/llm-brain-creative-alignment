import argparse
import csv
import os
from pathlib import Path
from statistics import mean, median, pstdev

from cadabra.utils import read_json, read_yaml


def resolve_path(path_value, model_dir=None):
    resolved_path = path_value
    if "${model_dir}" in resolved_path and not model_dir:
        raise ValueError(
            "The config path contains ${model_dir}, but no model directory was provided. "
            "Pass --model-dir or set the MODEL_DIR environment variable."
        )
    if model_dir:
        resolved_path = resolved_path.replace("${model_dir}", model_dir)
    resolved_path = os.path.expandvars(resolved_path)
    return Path(resolved_path)


def compute_length_stats(lengths):
    if not lengths:
        return None

    stats = {
        "count": len(lengths),
        "mean": mean(lengths),
        "median": median(lengths),
        "min": min(lengths),
        "max": max(lengths),
    }
    stats["std"] = pstdev(lengths) if len(lengths) > 1 else 0.0
    return stats


def load_output_lengths(datapath):
    data = read_json(datapath)
    items = data.get("data", [])

    lengths = []
    skipped = 0
    for item in items:
        output = item.get("output")
        if not isinstance(output, str):
            skipped += 1
            continue
        lengths.append(len(output.split()))

    return lengths, skipped, len(items)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize generation lengths for model datapaths in a YAML config."
    )
    parser.add_argument("--yaml-path", type=str, required=True, help="Path to the YAML config file.")
    parser.add_argument(
        "--datapath-name",
        type=str,
        required=True,
        help="Name of the datapath to match inside each model entry.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Value used to resolve ${model_dir} placeholders in the YAML.",
    )

    args = parser.parse_args()

    yaml_path = Path(args.yaml_path)
    alignment_data = read_yaml(yaml_path)
    model_dir = args.model_dir or os.environ.get("MODEL_DIR")
    output_dir = Path(__file__).resolve().parents[2] / "experiments" / "templeton_aut" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"generation_length_stats_{args.datapath_name}.csv"

    rows = []

    for model in alignment_data.get("model_data", []):
        model_name = model.get("name", "unknown")
        matching_datapaths = [
            datapath_entry
            for datapath_entry in model.get("datapaths", [])
            if datapath_entry.get("name") == args.datapath_name
        ]

        if not matching_datapaths:
            continue

        datapath_entry = matching_datapaths[0]
        datapath = resolve_path(datapath_entry["datapath"], model_dir=model_dir)

        if not datapath.exists():
            rows.append(
                {
                    "model_name": model_name,
                    "datapath_name": args.datapath_name,
                    "datapath": str(datapath),
                    "items": 0,
                    "usable_outputs": 0,
                    "skipped_items": 0,
                    "count": 0,
                    "mean": "",
                    "median": "",
                    "std": "",
                    "min": "",
                    "max": "",
                    "status": "missing_datapath",
                }
            )
            continue

        lengths, skipped, total_items = load_output_lengths(datapath)
        stats = compute_length_stats(lengths)

        row = {
            "model_name": model_name,
            "datapath_name": args.datapath_name,
            "datapath": str(datapath),
            "items": total_items,
            "usable_outputs": len(lengths),
            "skipped_items": skipped,
            "count": stats["count"] if stats else 0,
            "mean": f"{stats['mean']:.2f}" if stats else "",
            "median": f"{stats['median']:.2f}" if stats else "",
            "std": f"{stats['std']:.2f}" if stats else "",
            "min": stats["min"] if stats else "",
            "max": stats["max"] if stats else "",
            "status": "ok" if stats else "no_valid_outputs",
        }
        rows.append(row)

    fieldnames = [
        "model_name",
        "datapath_name",
        "datapath",
        "items",
        "usable_outputs",
        "skipped_items",
        "count",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "status",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generation length summary for datapath name: {args.datapath_name}")
    print(f"CSV saved to: {csv_path}")
    print()

    if not rows:
        print("No model datapaths matched the requested name.")
        return

    markdown_headers = [
        "model_name",
        # "items",
        # "usable_outputs",
        # "skipped_items",
        # "count",
        "mean",
        "median",
        "std",
        # "min",
        # "max",
        # "status",
    ]
    print("| " + " | ".join(markdown_headers) + " |")
    print("| " + " | ".join(["---"] * len(markdown_headers)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row["model_name"]),
                    # str(row["items"]),
                    # str(row["usable_outputs"]),
                    # str(row["skipped_items"]),
                    # str(row["count"]),
                    str(row["mean"]),
                    str(row["median"]),
                    str(row["std"]),
                    # str(row["min"]),
                    # str(row["max"]),
                    # str(row["status"]),
                ]
            )
            + " |"
        )


if __name__ == "__main__":
    main()