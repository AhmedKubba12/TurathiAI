"""Dataset of (image, caption, mask) triples for masked LoRA training.

Files that share a base name form one sample (Section 3.1):

    NAME.jpg              building image
    NAME-masklabel.png    segmentation mask
    NAME.txt              caption / annotation

When ``use_masked_loss`` is enabled the mask is returned alongside the image and,
downsampled to the latent grid, restricts the diffusion loss to building pixels
(see :func:`turathiai.training.losses.masked_mse_loss`).
"""
from __future__ import annotations

import collections
import glob
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


def index_triples(dataset_dir: str, mask_suffix: str = "-masklabel.png",
                  caption_ext: str = ".txt",
                  image_exts: tuple[str, ...] = ("jpg", "jpeg", "png")) -> list[dict]:
    """Group files into complete image/mask/caption triples."""
    def base_name(fn: str) -> str:
        low = fn.lower()
        if low.endswith(mask_suffix):
            return fn[: -len(mask_suffix)]
        return os.path.splitext(fn)[0]

    samples: dict[str, dict] = collections.defaultdict(
        lambda: {"image": None, "mask": None, "caption": None})
    for path in glob.glob(os.path.join(dataset_dir, "**", "*"), recursive=True):
        if not os.path.isfile(path):
            continue
        fn = os.path.basename(path)
        low = fn.lower()
        base = base_name(fn)
        if low.endswith(mask_suffix):
            samples[base]["mask"] = path
        elif low.endswith(caption_ext):
            samples[base]["caption"] = path
        elif any(low.endswith("." + e) for e in image_exts):
            samples[base]["image"] = path

    triples = [
        {"sample": b, **v}
        for b, v in sorted(samples.items())
        if v["image"] and v["mask"] and v["caption"]
    ]
    return triples


def train_val_split(triples: list[dict], val_split: float, seed: int = 42):
    """Deterministic split into train / validation lists."""
    rng = random.Random(seed)
    order = triples[:]
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_split)))
    return order[n_val:], order[:n_val]


class HeritageTriplesDataset(Dataset):
    """Yields pixel tensors, masks and raw caption strings.

    Text tokenisation / encoding is left to the training loop so the same dataset
    works across backbones with different tokenizers.
    """

    def __init__(self, triples: list[dict], resolution: int = 1024,
                 random_flip: bool = True, return_mask: bool = True):
        self.triples = triples
        self.resolution = resolution
        self.random_flip = random_flip
        self.return_mask = return_mask
        self.image_tf = T.Compose([
            T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
            T.CenterCrop(resolution),
        ])
        self.mask_tf = T.Compose([
            T.Resize(resolution, interpolation=T.InterpolationMode.NEAREST),
            T.CenterCrop(resolution),
        ])

    def __len__(self) -> int:
        return len(self.triples)

    def _read_caption(self, path: str) -> str:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            return fh.read().strip()

    def __getitem__(self, idx: int) -> dict:
        rec = self.triples[idx]
        image = self.image_tf(Image.open(rec["image"]).convert("RGB"))
        flip = self.random_flip and random.random() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # normalise to [-1, 1] for the VAE encoder
        px = (T.ToTensor()(image) * 2.0) - 1.0

        out = {"pixel_values": px, "caption": self._read_caption(rec["caption"]),
               "sample": rec["sample"]}

        if self.return_mask:
            mask = self.mask_tf(Image.open(rec["mask"]).convert("L"))
            if flip:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            m = torch.from_numpy((np.asarray(mask) > 127).astype(np.float32))[None]
            out["mask"] = m
        return out
