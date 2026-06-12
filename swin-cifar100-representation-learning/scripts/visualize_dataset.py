import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from swin_cifar100_rep.data import build_transform, get_dataloaders, get_raw_cifar100
from swin_cifar100_rep.plots import plot_class_distribution, plot_image_grid
from swin_cifar100_rep.utils import mkdir, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description='Visualize CIFAR-100 samples')
    p.add_argument('--data-dir', type=str, default='data')
    p.add_argument('--output-dir', type=str, default='outputs/plots')
    p.add_argument('--image-size', type=int, default=32)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    out = mkdir(args.output_dir)
    loaders = get_dataloaders(args.data_dir, args.image_size, args.batch_size, num_workers=args.num_workers, seed=args.seed)
    images, labels = next(iter(loaders['train']))
    plot_image_grid(images, labels.tolist(), loaders['class_names'], out / 'sample_grid.png')
    raw = get_raw_cifar100(args.data_dir, train=True, download=True)
    plot_class_distribution(raw.targets, raw.classes, out / 'class_distribution.png')
    print(f'Saved plots to {out}')


if __name__ == '__main__':
    main()
