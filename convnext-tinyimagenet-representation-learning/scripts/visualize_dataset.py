import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from convnext_tinyimagenet.data import TinyImageNetDataset, build_dataloaders, build_transform
from convnext_tinyimagenet.plots import plot_class_distribution, plot_image_grid
from convnext_tinyimagenet.utils import mkdir, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Tiny ImageNet samples and distribution")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="outputs/plots")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    output_dir = mkdir(args.output_dir)

    loaders = build_dataloaders(
        data_dir=args.data_dir,
        image_size=args.image_size,
        task="classification",
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    class_names = loaders["class_names"]

    images, labels = next(iter(loaders["train"]))
    plot_image_grid(
        images,
        labels.tolist(),
        class_names,
        str(output_dir / "sample_grid.png"),
        title="Tiny ImageNet training samples",
        max_images=36,
    )

    train_set = TinyImageNetDataset(
        data_dir=args.data_dir,
        split="train",
        transform=build_transform(args.image_size, train=False),
    )
    labels_for_distribution = [target for _, target in train_set.samples]
    plot_class_distribution(
        labels_for_distribution,
        train_set.classes,
        str(output_dir / "class_distribution.png"),
        title="Tiny ImageNet train class distribution",
    )

    print(f"Saved dataset plots to: {output_dir}")


if __name__ == "__main__":
    main()
