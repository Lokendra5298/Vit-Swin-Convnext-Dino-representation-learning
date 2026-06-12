import argparse
import sys
from pathlib import Path

from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from vit_cifar100.data import build_transforms
from vit_cifar100.model import VisionTransformer
from vit_cifar100.utils import get_device, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Predict CIFAR-100 class for one image')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--top-k', type=int, default=5)
    return parser.parse_args()


def build_model_from_checkpoint(checkpoint) -> VisionTransformer:
    args = checkpoint.get('args', {})
    return VisionTransformer(
        image_size=int(args.get('image_size', 32)),
        patch_size=int(args.get('patch_size', 4)),
        in_channels=3,
        num_classes=100,
        embed_dim=int(args.get('embed_dim', 192)),
        depth=int(args.get('depth', 6)),
        num_heads=int(args.get('num_heads', 6)),
        mlp_ratio=float(args.get('mlp_ratio', 4.0)),
        dropout=float(args.get('dropout', 0.0)),
        attention_dropout=float(args.get('attention_dropout', 0.0)),
        drop_path=float(args.get('drop_path', 0.0)),
    )


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device()

    checkpoint = load_checkpoint(args.checkpoint, map_location=str(device))
    class_names = checkpoint.get('class_names')
    if class_names is None:
        raise ValueError('Checkpoint does not contain class_names.')

    model = build_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    image_size = int(checkpoint.get('args', {}).get('image_size', 32))
    transform = build_transforms(image_size=image_size, train=False)

    image = Image.open(args.image).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = logits.softmax(dim=1)
    top_probs, top_indices = probs.topk(args.top_k, dim=1)

    print(f'Image: {args.image}')
    print(f'Top-{args.top_k} predictions:')
    for rank in range(args.top_k):
        idx = int(top_indices[0, rank])
        prob = float(top_probs[0, rank]) * 100
        print(f'{rank + 1}. {class_names[idx]}: {prob:.2f}%')


if __name__ == '__main__':
    main()
