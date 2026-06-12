from typing import Callable, List, Optional, Tuple

from PIL import Image, ImageOps
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


STL10_CLASSES = (
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
)

STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)


class Solarization:
    def __init__(self, p: float = 0.2):
        self.p = p

    def __call__(self, image: Image.Image) -> Image.Image:
        if torch.rand(1).item() < self.p:
            return ImageOps.solarize(image)
        return image


class GaussianBlur:
    def __init__(self, p: float = 1.0, kernel_size: int = 9, sigma: Tuple[float, float] = (0.1, 2.0)):
        self.p = p
        self.transform = transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def __call__(self, image: Image.Image) -> Image.Image:
        if torch.rand(1).item() < self.p:
            return self.transform(image)
        return image


class MultiCropTransform:
    """Return two global crops and N local crops for DINO-style training."""

    def __init__(self, global_size: int = 96, local_size: int = 48, local_crops_number: int = 6):
        normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(STL10_MEAN, STL10_STD)])
        color_jitter = transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8,
        )
        self.global_transform1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                color_jitter,
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0, kernel_size=9),
                normalize,
            ]
        )
        self.global_transform2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(global_size, scale=(0.4, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                color_jitter,
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1, kernel_size=9),
                Solarization(p=0.2),
                normalize,
            ]
        )
        self.local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(local_size, scale=(0.05, 0.4), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                color_jitter,
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.5, kernel_size=5),
                normalize,
            ]
        )
        self.local_crops_number = int(local_crops_number)

    def __call__(self, image: Image.Image) -> List[torch.Tensor]:
        crops = [self.global_transform1(image), self.global_transform2(image)]
        crops.extend([self.local_transform(image) for _ in range(self.local_crops_number)])
        return crops


def build_eval_transform(image_size: int = 96):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(STL10_MEAN, STL10_STD),
        ]
    )


def build_train_transform(image_size: int = 96):
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)],
                p=0.5,
            ),
            transforms.ToTensor(),
            transforms.Normalize(STL10_MEAN, STL10_STD),
        ]
    )


def get_ssl_dataset(
    data_dir: str = "data",
    global_size: int = 96,
    local_size: int = 48,
    local_crops_number: int = 6,
    download: bool = True,
    max_samples: Optional[int] = None,
) -> Dataset:
    dataset = datasets.STL10(
        root=data_dir,
        split="unlabeled",
        download=download,
        transform=MultiCropTransform(global_size, local_size, local_crops_number),
    )
    if max_samples is not None:
        dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return dataset


def get_labeled_dataset(
    data_dir: str = "data",
    split: str = "train",
    image_size: int = 96,
    train_transform: bool = False,
    download: bool = True,
    max_samples: Optional[int] = None,
) -> Dataset:
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")
    transform = build_train_transform(image_size) if train_transform else build_eval_transform(image_size)
    dataset = datasets.STL10(root=data_dir, split=split, download=download, transform=transform)
    if max_samples is not None:
        dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
    return dataset


def get_ssl_loader(
    data_dir: str = "data",
    global_size: int = 96,
    local_size: int = 48,
    local_crops_number: int = 6,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
    max_samples: Optional[int] = None,
):
    dataset = get_ssl_dataset(data_dir, global_size, local_size, local_crops_number, download, max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        drop_last=True,
    )


def get_labeled_loaders(
    data_dir: str = "data",
    image_size: int = 96,
    batch_size: int = 256,
    num_workers: int = 4,
    download: bool = True,
    train_aug: bool = False,
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
):
    train_set = get_labeled_dataset(data_dir, "train", image_size, train_aug, download, max_train_samples)
    test_set = get_labeled_dataset(data_dir, "test", image_size, False, download, max_test_samples)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=train_aug,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    return {
        "train": train_loader,
        "test": test_loader,
        "class_names": STL10_CLASSES,
        "train_dataset": train_set,
        "test_dataset": test_set,
    }
