import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from swin_cifar100_rep.data import get_dataloaders
from swin_cifar100_rep.model import SwinTransformer
from swin_cifar100_rep.utils import get_device, load_checkpoint, mkdir


def parse_tuple(text: str):
    return tuple(int(x.strip()) for x in text.split(',') if x.strip())


def build_model_from_args(args_dict):
    return SwinTransformer(
        image_size=int(args_dict.get('image_size', 32)),
        patch_size=int(args_dict.get('patch_size', 2)),
        num_classes=100,
        embed_dim=int(args_dict.get('embed_dim', 64)),
        depths=parse_tuple(str(args_dict.get('depths', '2,2,6,2'))),
        num_heads=parse_tuple(str(args_dict.get('num_heads', '2,4,8,16'))),
        window_size=int(args_dict.get('window_size', 4)),
        mlp_ratio=float(args_dict.get('mlp_ratio', 4.0)),
        dropout=float(args_dict.get('dropout', 0.0)),
        attn_dropout=float(args_dict.get('attn_dropout', 0.0)),
        drop_path=float(args_dict.get('drop_path', 0.0)),
        projection_dim=int(args_dict.get('projection_dim', 128)),
    )


def parse_args():
    p = argparse.ArgumentParser(description='Extract Swin features from CIFAR-100')
    p.add_argument('--data-dir', type=str, default='data')
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--output-dir', type=str, default='outputs/features')
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--num-workers', type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    features, labels = [], []
    for images, targets in tqdm(loader, leave=False):
        images = images.to(device)
        feat = model.forward_features(images).cpu().numpy()
        features.append(feat)
        labels.append(targets.numpy())
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def main():
    args = parse_args()
    out_dir = mkdir(args.output_dir)
    device = get_device()
    ckpt = load_checkpoint(args.checkpoint, map_location=str(device))
    model_args = ckpt.get('args', {})
    class_names = np.asarray(ckpt['class_names'])
    model = build_model_from_args(model_args).to(device)
    model.load_state_dict(ckpt['model'])

    loaders = get_dataloaders(
        data_dir=args.data_dir,
        image_size=int(model_args.get('image_size', 32)),
        batch_size=args.batch_size,
        val_ratio=float(model_args.get('val_ratio', 0.1)),
        num_workers=args.num_workers,
        task='classification',
        seed=int(model_args.get('seed', 42)),
    )

    for split in args.splits:
        feats, labs = extract(model, loaders[split], device)
        path = Path(out_dir) / f'{split}_features.npz'
        np.savez_compressed(path, features=feats, labels=labs, class_names=class_names)
        print(f'Saved {split} features: {path} shape={feats.shape}')


if __name__ == '__main__':
    main()
