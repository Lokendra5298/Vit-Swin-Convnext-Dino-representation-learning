import argparse
from pathlib import Path

import numpy as np

from swin_cifar100_rep.plots import plot_pca_embeddings


def parse_args():
    p = argparse.ArgumentParser(description='Visualize extracted features using PCA')
    p.add_argument('--features', type=str, required=True)
    p.add_argument('--output', type=str, default='outputs/plots/pca_embeddings.png')
    p.add_argument('--max-points', type=int, default=3000)
    return p.parse_args()


def main():
    args = parse_args()
    data = np.load(args.features, allow_pickle=True)
    plot_pca_embeddings(data['features'], data['labels'], data['class_names'], args.output, max_points=args.max_points)
    print(f'Saved PCA plot to: {args.output}')


if __name__ == '__main__':
    main()
