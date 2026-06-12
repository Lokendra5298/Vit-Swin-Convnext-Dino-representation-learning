import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from convnext_tinyimagenet.data import TinyImageNetDataset, build_transform
from convnext_tinyimagenet.model import create_convnext
from convnext_tinyimagenet.utils import get_device, load_checkpoint, mkdir


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ConvNeXt features from Tiny ImageNet")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs/features")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def build_model_from_checkpoint(checkpoint):
    args = checkpoint.get("args", {})
    return create_convnext(
        model_size=args.get("model_size", "nano"),
        num_classes=200,
        drop_path_rate=float(args.get("drop_path_rate", 0.0)),
        layer_scale_init_value=float(args.get("layer_scale_init_value", 1e-6)),
        head_dropout=float(args.get("head_dropout", 0.0)),
    )


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    features_list = []
    labels_list = []

    for images, labels in tqdm(loader, desc="extract"):
        images = images.to(device, non_blocking=True)
        features = model.forward_features(images)
        features_list.append(features.cpu().numpy())
        labels_list.append(labels.numpy())

    features = np.concatenate(features_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return features, labels


def main():
    args = parse_args()
    output_dir = mkdir(args.output_dir)

    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    class_names = checkpoint["class_names"]

    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    image_size = int(checkpoint.get("args", {}).get("image_size", 64))
    transform = build_transform(image_size=image_size, train=False)

    for split in args.splits:
        dataset = TinyImageNetDataset(args.data_dir, split=split, transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        features, labels = extract(model, loader, device)
        out_path = output_dir / f"{split}_features.npz"
        np.savez_compressed(
            out_path,
            features=features,
            labels=labels,
            class_names=np.array(class_names),
        )
        print(f"Saved {split} features: {out_path} with shape {features.shape}")


if __name__ == "__main__":
    main()
