import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from vit_cifar100.data import build_transforms, get_dataloaders, get_raw_cifar100
from vit_cifar100.plots import plot_augmentation_grid, plot_class_distribution, plot_image_grid
from vit_cifar100.utils import mkdir, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description='Create CIFAR-100 dataset visualizations')
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--output-dir', type=str, default='outputs/plots')
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rand-augment', action='store_true')
    parser.add_argument('--random-erasing', type=float, default=0.0)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    output_dir = mkdir(args.output_dir)

    loaders = get_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rand_augment=args.rand_augment,
        random_erasing=args.random_erasing,
        seed=args.seed,
    )
    class_names = loaders['class_names']

    images, labels = next(iter(loaders['train']))
    plot_image_grid(images, labels.tolist(), class_names, str(output_dir / 'sample_grid.png'), title='CIFAR-100 training samples', max_images=36)

    raw_train = get_raw_cifar100(args.data_dir, train=True, download=True)
    plot_class_distribution(raw_train.targets, raw_train.classes, str(output_dir / 'class_distribution.png'), title='CIFAR-100 class distribution')

    train_transform = build_transforms(image_size=args.image_size, train=True, rand_augment=args.rand_augment, random_erasing=args.random_erasing)
    original_image, _ = raw_train[0]
    plot_augmentation_grid(original_image, train_transform, str(output_dir / 'augmentation_grid.png'), title='Random augmentations of one training image', n=12)

    print(f'Saved dataset visualizations to: {output_dir}')


if __name__ == '__main__':
    main()
