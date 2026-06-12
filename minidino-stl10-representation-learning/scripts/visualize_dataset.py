import argparse
import sys
from pathlib import Path

import torch
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from minidino_stl10.data import STL10_CLASSES, MultiCropTransform, build_eval_transform, get_labeled_loaders
from minidino_stl10.plots import plot_image_grid, plot_multicrop_grid
from minidino_stl10.utils import mkdir, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize STL-10 and DINO multi-crop views")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/plots")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--global-size", type=int, default=96)
    parser.add_argument("--local-size", type=int, default=48)
    parser.add_argument("--local-crops-number", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=36)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = mkdir(args.output_dir)

    loaders = get_labeled_loaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_aug=False,
    )

    train_images, train_labels = next(iter(loaders["train"]))
    plot_image_grid(
        train_images,
        train_labels.tolist(),
        STL10_CLASSES,
        str(output_dir / "stl10_labeled_samples.png"),
        title="STL-10 labeled train samples",
    )

    unlabeled = datasets.STL10(root=args.data_dir, split="unlabeled", download=True, transform=build_eval_transform(args.image_size))
    unlabeled_images = []
    for idx in range(min(args.batch_size, len(unlabeled))):
        image, _ = unlabeled[idx]
        unlabeled_images.append(image)

    plot_image_grid(
        torch.stack(unlabeled_images),
        None,
        STL10_CLASSES,
        str(output_dir / "stl10_unlabeled_samples.png"),
        title="STL-10 unlabeled samples",
    )

    raw_unlabeled = datasets.STL10(root=args.data_dir, split="unlabeled", download=True)
    image, _ = raw_unlabeled[0]
    transform = MultiCropTransform(args.global_size, args.local_size, args.local_crops_number)
    crops = transform(image)
    plot_multicrop_grid(crops, str(output_dir / "multicrop_views.png"))

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()
