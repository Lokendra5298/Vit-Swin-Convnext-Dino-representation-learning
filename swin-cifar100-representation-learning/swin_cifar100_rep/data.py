from pathlib import Path
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class Identity:
    def __call__(self, x):
        return x


class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


def build_transform(image_size: int = 32, train: bool = True, strong: bool = False):
    resize = Identity() if image_size == 32 else transforms.Resize((image_size, image_size))

    if train:
        ops = [
            resize,
            transforms.RandomCrop(image_size, padding=max(4, image_size // 8), padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),
        ]
        if strong:
            ops.extend([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.2),
            ])
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ])
        return transforms.Compose(ops)

    return transforms.Compose([
        resize,
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])


def _limit_dataset(dataset: Dataset, max_samples: Optional[int]) -> Dataset:
    if max_samples is None:
        return dataset
    return Subset(dataset, list(range(min(max_samples, len(dataset)))))


def get_datasets(
    data_dir: str = 'data',
    image_size: int = 32,
    val_ratio: float = 0.1,
    task: str = 'classification',
    seed: int = 42,
    download: bool = True,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
):
    data_dir = str(Path(data_dir))
    strong = task == 'supcon'
    train_transform = build_transform(image_size=image_size, train=True, strong=strong)
    if task == 'supcon':
        train_transform = TwoCropTransform(train_transform)

    train_full = datasets.CIFAR100(root=data_dir, train=True, download=download, transform=train_transform)
    val_size = int(len(train_full) * val_ratio)
    train_size = len(train_full) - val_size
    generator = torch.Generator().manual_seed(seed)
    train_set, val_indices_subset = random_split(train_full, [train_size, val_size], generator=generator)

    val_full = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=download,
        transform=build_transform(image_size=image_size, train=False),
    )
    val_set = Subset(val_full, val_indices_subset.indices)

    test_set = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=download,
        transform=build_transform(image_size=image_size, train=False),
    )

    train_set = _limit_dataset(train_set, max_train_samples)
    val_set = _limit_dataset(val_set, max_val_samples)
    test_set = _limit_dataset(test_set, max_test_samples)
    return train_set, val_set, test_set, tuple(train_full.classes)


def get_dataloaders(
    data_dir: str = 'data',
    image_size: int = 32,
    batch_size: int = 128,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    task: str = 'classification',
    seed: int = 42,
    download: bool = True,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
):
    train_set, val_set, test_set, class_names = get_datasets(
        data_dir=data_dir,
        image_size=image_size,
        val_ratio=val_ratio,
        task=task,
        seed=seed,
        download=download,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
    )

    pin_memory = torch.cuda.is_available()
    common = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **common)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, **common)

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'class_names': class_names,
        'train_dataset': train_set,
        'val_dataset': val_set,
        'test_dataset': test_set,
    }


def get_raw_cifar100(data_dir: str = 'data', train: bool = True, download: bool = True):
    return datasets.CIFAR100(root=data_dir, train=train, download=download)
