"""Automated building segmentation with the Segment Anything Model (SAM).

Pipeline (Section 3.1.3 of the paper):

1.  Downscale each building photo to 1024x1024 (memory / time constraints).
2.  Run SAM's automatic mask generator to obtain many candidate masks.
3.  Rank the candidates with the heuristic scoring function (Equation 1) and
    keep the highest-scoring one (large, compact, connected -> a building).
4.  Refine the mask (morphological closing + largest contour) and upscale it
    back to the image's native resolution.
5.  Save it next to the image as ``<NAME>-masklabel.png``.

The SAM checkpoint (e.g. ``sam_vit_h_4b8939.pth``) must be downloaded from the
official Segment Anything repository; see ``data/README.md``.

Usage
-----
    python -m turathiai.segmentation.sam_segment \
        --images-dir data/CHDB_Full \
        --sam-checkpoint checkpoints/sam_vit_h_4b8939.pth \
        --model-type vit_h
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

from .postprocess import refine_mask
from .scoring import bbox_fill_ratio, mask_score, select_best_mask

WORKING_RESOLUTION = 1024
MASK_SUFFIX = "-masklabel.png"
IMAGE_EXTS = ("jpg", "jpeg", "png")


def _load_sam_generator(sam_checkpoint: str, model_type: str, device: str,
                        points_per_side: int = 32):
    """Build a SAM automatic mask generator. Imported lazily so the rest of the
    package works without ``segment-anything`` installed."""
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device if torch.cuda.is_available() else "cpu")
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.90,
        min_mask_region_area=1000,
    )


def find_images(images_dir: str) -> list[str]:
    """List building images, excluding existing mask files."""
    paths: list[str] = []
    for ext in IMAGE_EXTS:
        paths += glob.glob(os.path.join(images_dir, "**", f"*.{ext}"), recursive=True)
        paths += glob.glob(os.path.join(images_dir, "**", f"*.{ext.upper()}"), recursive=True)
    paths = [p for p in paths if MASK_SUFFIX.split(".")[0] not in os.path.basename(p).lower()]
    return sorted(set(paths))


def segment_image(generator, image_bgr: np.ndarray,
                  working_resolution: int = WORKING_RESOLUTION) -> np.ndarray:
    """Generate, score, select and refine the mask for a single image.

    Returns a uint8 {0, 255} mask at the image's native resolution.
    """
    native_h, native_w = image_bgr.shape[:2]
    small = cv2.resize(image_bgr, (working_resolution, working_resolution),
                       interpolation=cv2.INTER_AREA)
    small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    records = generator.generate(small_rgb)
    candidates = [r["segmentation"].astype(bool) for r in records]
    if not candidates:
        return np.zeros((native_h, native_w), dtype=np.uint8)

    best_idx, _ = select_best_mask(candidates)
    return refine_mask(candidates[best_idx], native_size=(native_w, native_h))


def run(images_dir: str, sam_checkpoint: str, model_type: str = "vit_h",
        device: str = "cuda", overwrite: bool = False) -> None:
    generator = _load_sam_generator(sam_checkpoint, model_type, device)
    images = find_images(images_dir)
    print(f"Found {len(images)} images under {images_dir}")

    for i, path in enumerate(images, 1):
        base, _ = os.path.splitext(path)
        out_path = base + MASK_SUFFIX
        if os.path.exists(out_path) and not overwrite:
            continue
        image = cv2.imread(path)
        if image is None:
            print(f"  [skip] could not read {path}")
            continue
        mask = segment_image(generator, image)
        cv2.imwrite(out_path, mask)
        cov = float((mask > 127).mean())
        fill = bbox_fill_ratio(mask > 127)
        print(f"[{i}/{len(images)}] {os.path.basename(path)}  "
              f"coverage={cov:.2f} bbox_fill={fill:.2f} -> {os.path.basename(out_path)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="SAM building segmentation with heuristic scoring.")
    ap.add_argument("--images-dir", required=True, help="Directory of building photos (recursive).")
    ap.add_argument("--sam-checkpoint", required=True, help="Path to the SAM checkpoint .pth.")
    ap.add_argument("--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true", help="Regenerate existing masks.")
    args = ap.parse_args()
    run(args.images_dir, args.sam_checkpoint, args.model_type, args.device, args.overwrite)


if __name__ == "__main__":
    main()
