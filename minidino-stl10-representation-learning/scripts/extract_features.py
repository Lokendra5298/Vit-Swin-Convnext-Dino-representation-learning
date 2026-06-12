import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torchvision import datasets

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from minidino_stl10.data import STL10_CLASSES, build_eval_transform
from minidino_stl10.model import create_dino_model
from minidino_stl10.utils import get_device, load_checkpoint, mkdir


def parse_args():
    parser = argparse.ArgumentParser(description="Extract features from Mini-DINO checkpoint")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="outputs_dino/features")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test", "unlabeled"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-teacher", action="store_true", help="Use teacher weights instead of student weights")
    return parser.parse_args()


def build_backbone_from_checkpoint(checkpoint, use_teacher: bool):
    args = checkpoint.get("args", {})
    model = create_dino_model(
        model_size=args.get("model_size", "small"),
        image_size=int(args.get("image_size", 96)),
        patch_size=int(args.get("patch_size", 8)),
        num_register_tokens=int(args.get("num_register_tokens", 4)),
        drop_path_rate=0.0,
        out_dim=int(args.get("out_dim", 4096)),
        hidden_dim=int(args.get("hidden_dim", 2048)),
        bottleneck_dim=int(args.get("bottleneck_dim", 256)),
    )
    state_key = "teacher" if use_teacher and "teacher" in checkpoint else "student"
    model.load_state_dict(checkpoint[state_key])
    return model.backbone


@torch.no_grad()
def extract(backbone, loader, device):
    backbone.eval()
    features_list = []
    labels_list = []
    for images, labels in tqdm(loader, desc="extract"):
        images = images.to(device, non_blocking=True)
        features = backbone.forward_features(images)
        features = torch.nn.functional.normalize(features, dim=1)
        features_list.append(features.cpu().numpy())
        labels_list.append(labels.numpy())
    return np.concatenate(features_list, axis=0), np.concatenate(labels_list, axis=0)


def main():
    args = parse_args()
    output_dir = mkdir(args.output_dir)
    device = get_device()
    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    backbone = build_backbone_from_checkpoint(checkpoint, use_teacher=args.use_teacher).to(device)

    image_size = int(checkpoint.get("args", {}).get("image_size", 96))
    transform = build_eval_transform(image_size)

    for split in args.splits:
        dataset = datasets.STL10(root=args.data_dir, split=split, download=True, transform=transform)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        features, labels = extract(backbone, loader, device)
        out_path = output_dir / f"{split}_features.npz"
        np.savez_compressed(out_path, features=features, labels=labels, class_names=np.array(STL10_CLASSES))
        print(f"Saved {split} features to {out_path}, shape={features.shape}")


if __name__ == "__main__":
    main()
