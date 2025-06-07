import os
import random
from glob import glob
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T


class FlowerDataset(Dataset):
    """Custom dataset for flower classification with triplet learning support."""

    def __init__(self, root: str, transformations: Optional[T.Compose] = None):
        """
        Initialize the flower dataset.

        Args:
            root: Path to the dataset directory
            transformations: Image transformations to apply
        """
        self.transformations = transformations
        self.im_paths = glob(f"{root}/*/*.jpg")
        self.cls_names: Dict[str, int] = {}
        self.cls_counts: Dict[str, int] = {}

        self._build_class_mappings()

    def _build_class_mappings(self) -> None:
        """Build class name to index mappings and count samples per class."""
        count = 0
        for im_path in self.im_paths:
            cls_name = self.get_cls_name(im_path)
            if cls_name not in self.cls_names:
                self.cls_names[cls_name] = count
                count += 1

            if cls_name not in self.cls_counts:
                self.cls_counts[cls_name] = 1
            else:
                self.cls_counts[cls_name] += 1

    def get_cls_name(self, path: str) -> str:
        """Extract class name from file path."""
        return os.path.dirname(path).split("/")[-1]

    def __len__(self) -> int:
        return len(self.im_paths)

    def get_pos_neg_im_paths(self, qry_label: str) -> Tuple[str, str]:
        """
        Get positive and negative sample paths for triplet learning.

        Args:
            qry_label: Query image class label

        Returns:
            Tuple of (positive_path, negative_path)
        """
        pos_im_paths = [
            im_path
            for im_path in self.im_paths
            if qry_label == self.get_cls_name(im_path)
        ]
        neg_im_paths = [
            im_path
            for im_path in self.im_paths
            if qry_label != self.get_cls_name(im_path)
        ]

        pos_rand_int = random.randint(0, len(pos_im_paths) - 1)
        neg_rand_int = random.randint(0, len(neg_im_paths) - 1)

        return pos_im_paths[pos_rand_int], neg_im_paths[neg_rand_int]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a triplet sample (query, positive, negative).

        Args:
            idx: Sample index

        Returns:
            Dictionary with query, positive, negative images and labels
        """
        im_path = self.im_paths[idx]
        qry_im = Image.open(im_path).convert("RGB")
        qry_label = self.get_cls_name(im_path)

        pos_im_path, neg_im_path = self.get_pos_neg_im_paths(qry_label)
        pos_im = Image.open(pos_im_path).convert("RGB")
        neg_im = Image.open(neg_im_path).convert("RGB")

        qry_gt = self.cls_names[qry_label]
        neg_gt = self.cls_names[self.get_cls_name(neg_im_path)]

        if self.transformations is not None:
            qry_im = self.transformations(qry_im)
            pos_im = self.transformations(pos_im)
            neg_im = self.transformations(neg_im)

        return {
            "qry_im": qry_im,
            "qry_gt": qry_gt,
            "pos_im": pos_im,
            "neg_im": neg_im,
            "neg_gt": neg_gt,
        }


def get_data_loaders(
    root: str,
    transformations: T.Compose,
    batch_size: int,
    split: List[float] = [0.9, 0.05, 0.05],
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int], Dict[str, int]]:
    """
    Create train, validation, and test data loaders.

    Args:
        root: Path to dataset directory
        transformations: Image transformations
        batch_size: Batch size for data loaders
        split: Train/validation/test split ratios
        num_workers: Number of workers for data loading

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names, class_counts)
    """
    dataset = FlowerDataset(root=root, transformations=transformations)
    total_len = len(dataset)

    train_len = int(total_len * split[0])
    val_len = int(total_len * split[1])
    test_len = total_len - (train_len + val_len)

    train_ds, val_ds, test_ds = random_split(
        dataset=dataset, lengths=[train_len, val_len, test_len]
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, dataset.cls_names, dataset.cls_counts


def get_default_transforms(size: int = 224) -> T.Compose:
    """
    Get default image transformations for flower classification.

    Args:
        size: Target image size

    Returns:
        Composed transformations
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return T.Compose(
        [
            T.ToTensor(),
            T.Resize(size=(size, size), antialias=False),
            T.Normalize(mean=mean, std=std),
        ]
    )
