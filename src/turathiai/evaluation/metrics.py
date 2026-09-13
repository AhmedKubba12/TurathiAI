"""Image-quality metrics (Section 3.3).

* **CLIP score** — text-image alignment via ``open_clip`` ViT-B-32 (openai).
  Computed independently of the transformers CLIP API for robustness (matches
  the evaluation notebook). Returned as ``100 * cosine_similarity``.
* **FID** — Frechet Inception Distance against the real residential dataset
  (2048-dim InceptionV3 pool features).
* **LPIPS / SSIM** — perceptual and structural distance of each model's outputs
  relative to the base SDXL model's outputs (AlexNet backbone for LPIPS).
"""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

_OC = {"m": None}


def _open_clip(device: str, model_name: str = "ViT-B-32", pretrained: str = "openai"):
    if _OC["m"] is None:
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)[2]
        tok = open_clip.get_tokenizer(model_name)
        _OC["m"] = (model.eval().to(device), preprocess, tok)
    return _OC["m"]


@torch.no_grad()
def clip_scores(records: list[dict], device: str = "cuda",
                model_name: str = "ViT-B-32", pretrained: str = "openai") -> np.ndarray:
    """Per-image CLIP score for a list of ``{"path", "prompt"}`` records."""
    model, preprocess, tok = _open_clip(device, model_name, pretrained)
    vals = []
    for rec in records:
        img = preprocess(Image.open(rec["path"]).convert("RGB")).unsqueeze(0).to(device)
        txt = tok([rec["prompt"]]).to(device)
        imf = model.encode_image(img)
        txf = model.encode_text(txt)
        imf = imf / imf.norm(dim=-1, keepdim=True)
        txf = txf / txf.norm(dim=-1, keepdim=True)
        vals.append((100.0 * (imf * txf).sum(-1)).clamp(min=0).item())
    return np.array(vals)


_to_uint8 = T.Compose([T.Resize((299, 299)), T.ToTensor()])


def _load_uint8(path: str) -> torch.Tensor:
    return (_to_uint8(Image.open(path).convert("RGB")) * 255).to(torch.uint8)


def fid_against_real(gen_records: list[dict], real_paths: list[str],
                     device: str = "cuda", feature: int = 2048, batch: int = 64) -> float:
    """FID between generated images and the real residential dataset."""
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid = FrechetInceptionDistance(feature=feature, normalize=False).to(device)
    for i in range(0, len(real_paths), batch):
        chunk = torch.stack([_load_uint8(p) for p in real_paths[i:i + batch]]).to(device)
        fid.update(chunk, real=True)
    for i in range(0, len(gen_records), batch):
        chunk = torch.stack([_load_uint8(g["path"]) for g in gen_records[i:i + batch]]).to(device)
        fid.update(chunk, real=False)
    return fid.compute().item()


def lpips_ssim_vs_base(gen_records: list[dict], base_records: list[dict],
                       device: str = "cuda", net: str = "alex"):
    """LPIPS (lower = closer) and SSIM (higher = closer) vs. the base model."""
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    lp = LearnedPerceptualImagePatchSimilarity(net_type=net, normalize=True).to(device)
    ss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    tf = T.Compose([T.Resize((512, 512)), T.ToTensor()])

    lv, sv = [], []
    for g, b in zip(gen_records, base_records):
        a = tf(Image.open(g["path"]).convert("RGB")).unsqueeze(0).to(device)
        c = tf(Image.open(b["path"]).convert("RGB")).unsqueeze(0).to(device)
        lv.append(lp(a, c).item())
        sv.append(ss(a, c).item())
    return np.array(lv), np.array(sv)
