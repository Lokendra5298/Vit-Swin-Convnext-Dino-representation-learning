import argparse
from pathlib import Path

from torchvision import datasets


def parse_args():
    parser = argparse.ArgumentParser(description="Download STL-10 train/test/unlabeled splits")
    parser.add_argument("--data-dir", type=str, default="data")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "test", "unlabeled"]:
        print(f"Downloading/checking STL-10 split: {split}")
        dataset = datasets.STL10(root=args.data_dir, split=split, download=True)
        print(f"  {split}: {len(dataset)} samples")

    print(f"Done. Dataset saved under: {data_dir}")


if __name__ == "__main__":
    main()
