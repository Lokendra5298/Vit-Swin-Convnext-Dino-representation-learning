import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Markdown report for Mini-DINO STL-10 experiments")
    parser.add_argument("--output-dir", type=str, default="outputs_dino")
    parser.add_argument("--linear-probe-dir", type=str, default="outputs_dino_linear_probe")
    return parser.parse_args()


def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def read_history(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    linear_dir = Path(args.linear_probe_dir)
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    pretrain_history = read_history(output_dir / "logs" / "history.csv")
    probe_metrics = read_json(linear_dir / "metrics" / "test_metrics.json")

    lines = []
    lines.append("# Mini-DINOv2-Style STL-10 Experiment Report\n")
    lines.append("This report summarizes self-supervised pretraining on STL-10 unlabeled data and downstream evaluation on STL-10 labeled train/test data.\n")
    lines.append("## Dataset and views\n")
    lines.append("![Unlabeled samples](../plots/stl10_unlabeled_samples.png)\n")
    lines.append("![Multi-crop views](../plots/multicrop_views.png)\n")
    lines.append("## Pretraining\n")
    lines.append("![Pretraining curves](../plots/pretraining_curves.png)\n")

    if pretrain_history:
        best = min(pretrain_history, key=lambda row: float(row["loss"]))
        lines.append(f"- Best pretraining epoch by loss: `{best['epoch']}`\n")
        lines.append(f"- Best DINO loss: `{float(best['loss']):.4f}`\n")

    lines.append("## Feature visualization\n")
    lines.append("![PCA embeddings](../plots/pca_embeddings.png)\n")
    lines.append("## Linear probe\n")

    if probe_metrics is not None:
        lines.append(f"- Test top-1 accuracy: `{probe_metrics['test_top1_percent']:.2f}%`\n")
        lines.append(f"- Test top-5 accuracy: `{probe_metrics['test_top5_percent']:.2f}%`\n")
        lines.append(f"- Macro F1: `{probe_metrics['macro_f1']:.4f}`\n")
        lines.append(f"- Weighted F1: `{probe_metrics['weighted_f1']:.4f}`\n")
    else:
        lines.append("Run `scripts/linear_probe.py` to add linear-probe metrics here.\n")

    lines.append("## Notes\n")
    lines.append("- This is a compact educational DINOv2-style implementation.\n")
    lines.append("- For stronger results, train longer, increase model size, and tune multi-crop settings.\n")
    lines.append("- Use k-NN evaluation and linear probing to measure representation quality.\n")

    report_path = report_dir / "minidino_stl10_report.md"
    report_path.write_text("\n".join(lines))
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
