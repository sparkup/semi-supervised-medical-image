"""Feature extraction utilities using pretrained CNN backbones."""

import os
from typing import Tuple
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
import numpy as np


def pick_device() -> torch.device:
    """Pick CUDA, Apple MPS or CPU depending on availability."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_transform(image_size: int = 224) -> transforms.Compose:
    """Standard ImageNet transform compatible with torchvision backbones."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_model(backbone: str = "resnet18", device: torch.device = None) -> nn.Module:
    """Build a pretrained backbone and strip its classification head to output embeddings."""
    if not hasattr(models, backbone):
        raise ValueError(f"Unknown backbone: {backbone}")

    ctor = getattr(models, backbone)

    # Modern weights API (torchvision >=0.13)
    weights = None
    try:
        weights_enum = getattr(models, f"{backbone.capitalize()}_Weights")
        weights = weights_enum.DEFAULT  # safest option
    except AttributeError:
        # If no weights enum exists, fallback = no pretrained weights
        print(f"[INFO] No weights enum found for {backbone}, using randomly initialized weights.")

    model = ctor(weights=weights)

    # Remove classification head to expose penultimate features
    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        model.fc = nn.Identity()
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        model.classifier = nn.Identity()

    model.eval()
    if device is not None:
        model.to(device)
    return model


def build_dataloader(
    data_labeled: str,
    batch_size: int,
    transform: transforms.Compose,
    return_dataset: bool = False
) -> DataLoader:
    """Create a deterministic dataloader for labeled data using ImageFolder.
    If return_dataset is True, return (loader, ds) tuple, else return only loader.
    """
    class ImageFolderWithPaths(datasets.ImageFolder):
        def __getitem__(self, index):
            original_tuple = super().__getitem__(index)
            path = self.samples[index][0]
            return original_tuple + (path,)

    ds = ImageFolderWithPaths(data_labeled, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
    if return_dataset:
        return loader, ds
    return loader


def extract_embeddings(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, list]:
    """Extract flattened embeddings, labels, and image paths from a dataloader."""
    feats_list, labels_list = [], []
    paths_list = []
    model.to(device)
    with torch.no_grad():
        for i, (x, y, path) in enumerate(loader):
            x = x.to(device)
            f = model(x)
            # Flatten (N, C, 1, 1) -> (N, C) or (N, C, H, W) -> (N, C*H*W)
            f = torch.flatten(f, 1)
            feats_list.append(f.cpu().numpy())
            labels_list.append(y.numpy())
            # Collect paths if possible
            paths_list.extend(path)
    features = np.concatenate(feats_list, axis=0) if feats_list else np.empty((0,), dtype=float)
    labels = np.concatenate(labels_list, axis=0) if labels_list else np.empty((0,), dtype=int)
    return features, labels, paths_list


def save_embeddings(emb_dir: str, features: np.ndarray, labels: np.ndarray, paths: list | None = None) -> None:
    """Persist features/labels in the embeddings directory."""
    os.makedirs(emb_dir, exist_ok=True)
    np.save(os.path.join(emb_dir, "features.npy"), features)
    np.save(os.path.join(emb_dir, "labels.npy"), labels)
    if paths is not None:
        np.save(os.path.join(emb_dir, "paths.npy"), np.array(paths))
