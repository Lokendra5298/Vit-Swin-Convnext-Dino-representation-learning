import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from convnext_tinyimagenet.plots import plot_pca_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize feature embeddings with PCA")
    parser.add_argument("--features", type=str, required=True, help="Path to .npz file from extract_features.py")
    parser.add_argument("--output", type=str, default="outputs/plots/pca_embeddings.png")
    parser.add_argument("--max-points", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    data = np.load(args.features, allow_pickle=True)

    features = data["features"]
    labels = data["labels"]
    class_names = data["class_names"].tolist()

    plot_pca_embeddings(
        features,
        labels,
        class_names,
        args.output,
        max_points=args.max_points,
        title="PCA visualization of ConvNeXt Tiny ImageNet features",
    )
    print(f"Saved PCA plot to: {args.output}")


if __name__ == "__main__":
    main()
