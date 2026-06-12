from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def find_tinyimagenet_root(data_dir: str) -> Path:
    data_dir = Path(data_dir)
    if (data_dir / "wnids.txt").exists() and (data_dir / "words.txt").exists():
        return data_dir
    candidate = data_dir / "tiny-imagenet-200"
    if (candidate / "wnids.txt").exists() and (candidate / "words.txt").exists():
        return candidate

    raise FileNotFoundError(
        "Could not find Tiny ImageNet. Expected either data_dir/wnids.txt "
        "or data_dir/tiny-imagenet-200/wnids.txt. Run scripts/prepare_tinyimagenet.py first."
    )


def read_wnids(root: Path) -> List[str]:
    with (root / "wnids.txt").open("r") as f:
        return [line.strip() for line in f if line.strip()]


def read_words(root: Path) -> Dict[str, str]:
    words = {}
    words_path = root / "words.txt"
    with words_path.open("r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                wnid = parts[0]
                label = parts[1].split(",")[0].strip()
                words[wnid] = label
    return words


class TinyImageNetDataset(Dataset):
    """Tiny ImageNet-200 dataset reader.

    This class reads the original Tiny ImageNet folder layout directly:
    - train labels come from train/<wnid>/images
    - val labels come from val/val_annotations.txt
    - test labels are unknown and returned as -1
    """

    def __init__(
        self,
        data_dir: str = "data",
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = find_tinyimagenet_root(data_dir)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of train, val, or test")

        self.wnids = read_wnids(self.root)
        self.class_to_idx = {wnid: idx for idx, wnid in enumerate(self.wnids)}
        self.words = read_words(self.root)
        self.classes = [self.words.get(wnid, wnid) for wnid in self.wnids]

        self.samples: List[Tuple[Path, int]] = self._make_samples()

    def _make_samples(self) -> List[Tuple[Path, int]]:
        samples: List[Tuple[Path, int]] = []

        if self.split == "train":
            train_dir = self.root / "train"
            for wnid in self.wnids:
                image_dir = train_dir / wnid / "images"
                if not image_dir.exists():
                    continue
                for path in sorted(image_dir.glob("*.JPEG")):
                    samples.append((path, self.class_to_idx[wnid]))

        elif self.split == "val":
            val_dir = self.root / "val"
            annot_path = val_dir / "val_annotations.txt"
            with annot_path.open("r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    filename, wnid = parts[0], parts[1]
                    image_path = val_dir / "images" / filename
                    samples.append((image_path, self.class_to_idx[wnid]))

        else:
            test_dir = self.root / "test" / "images"
            for path in sorted(test_dir.glob("*.JPEG")):
                samples.append((path, -1))

        if not samples:
            raise RuntimeError(f"No samples found for split={self.split} at {self.root}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target


class TwoCropsTransform:
    """Create two independently augmented views of the same image."""

    def __init__(self, base_transform: Callable):
        self.base_transform = base_transform

    def __call__(self, image):
        return self.base_transform(image), self.base_transform(image)


def build_transform(
    image_size: int = 64,
    train: bool = True,
    rand_augment: bool = True,
    random_erasing: float = 0.0,
    supcon: bool = False,
):
    if train and supcon:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.4,
                            hue=0.1,
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    if train:
        transform_list = [
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]

        if rand_augment and hasattr(transforms, "RandAugment"):
            transform_list.append(transforms.RandAugment(num_ops=2, magnitude=9))

        transform_list.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        if random_erasing > 0:
            transform_list.append(transforms.RandomErasing(p=random_erasing))

        return transforms.Compose(transform_list)

    return transforms.Compose(
        [
            transforms.Resize(image_size + 8),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_datasets(
    data_dir: str = "data",
    image_size: int = 64,
    task: str = "classification",
    rand_augment: bool = True,
    random_erasing: float = 0.0,
):
    if task not in {"classification", "supcon"}:
        raise ValueError("task must be classification or supcon")

    if task == "supcon":
        train_transform = TwoCropsTransform(
            build_transform(image_size=image_size, train=True, supcon=True)
        )
    else:
        train_transform = build_transform(
            image_size=image_size,
            train=True,
            rand_augment=rand_augment,
            random_erasing=random_erasing,
            supcon=False,
        )

    eval_transform = build_transform(image_size=image_size, train=False)

    train_set = TinyImageNetDataset(data_dir=data_dir, split="train", transform=train_transform)
    val_set = TinyImageNetDataset(data_dir=data_dir, split="val", transform=eval_transform)

    return train_set, val_set, train_set.classes


def build_dataloaders(
    data_dir: str = "data",
    image_size: int = 64,
    task: str = "classification",
    batch_size: int = 128,
    num_workers: int = 4,
    rand_augment: bool = True,
    random_erasing: float = 0.0,
):
    train_set, val_set, class_names = build_datasets(
        data_dir=data_dir,
        image_size=image_size,
        task=task,
        rand_augment=rand_augment,
        random_erasing=random_erasing,
    )

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=task == "supcon",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "class_names": class_names,
        "train_dataset": train_set,
        "val_dataset": val_set,
    }
