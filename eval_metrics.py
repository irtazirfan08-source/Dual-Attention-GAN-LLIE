"""
Standalone Quantitative Evaluation CLI (PSNR & SSIM) for Dual-Attention-GAN-LLIE.
Computes image restoration metrics across paired enhanced and ground-truth directories
without modifying core thesis architecture or training scripts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _to_bchw_float01(image: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Convert HWC/CHW/BCHW image array or tensor into [B, C, H, W] float32 in [0, 1]."""
    if isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 2:
            tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        elif arr.ndim == 3:
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        elif arr.ndim == 4:
            tensor = torch.from_numpy(arr).permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported NumPy image shape: {image.shape}")
        return tensor.clamp(0.0, 1.0)

    if isinstance(image, torch.Tensor):
        tensor = image.detach().float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 4:
            raise ValueError(f"Unsupported Tensor image shape: {tuple(image.shape)}")
        return tensor.clamp(0.0, 1.0)

    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def compute_psnr(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    max_val: float = 1.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (dB) between predicted and target images."""
    pred_t = _to_bchw_float01(pred)
    target_t = _to_bchw_float01(target)
    if pred_t.shape != target_t.shape:
        raise ValueError(f"Shape mismatch: {tuple(pred_t.shape)} vs {tuple(target_t.shape)}")

    mse = torch.mean((pred_t - target_t) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return round(10.0 * math.log10((max_val ** 2) / mse), 4)


def _gaussian_window(window_size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - (window_size // 2)
    gauss = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    gauss = gauss / gauss.sum()
    kernel_2d = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    return kernel_2d.expand(channels, 1, window_size, window_size).contiguous()


def compute_ssim(
    pred: np.ndarray | torch.Tensor,
    target: np.ndarray | torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Compute Structural Similarity Index (SSIM) using an 11x11 Gaussian window."""
    pred_t = _to_bchw_float01(pred)
    target_t = _to_bchw_float01(target)
    if pred_t.shape != target_t.shape:
        raise ValueError(f"Shape mismatch: {tuple(pred_t.shape)} vs {tuple(target_t.shape)}")

    _, channels, height, width = pred_t.shape
    effective_window = min(window_size, height, width)
    if effective_window % 2 == 0:
        effective_window = max(1, effective_window - 1)

    window = _gaussian_window(effective_window, sigma, channels, pred_t.device)
    pad = effective_window // 2

    mu1 = F.conv2d(pred_t, window, padding=pad, groups=channels)
    mu2 = F.conv2d(target_t, window, padding=pad, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred_t * pred_t, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(target_t * target_t, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(pred_t * target_t, window, padding=pad, groups=channels) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    numerator = (2.0 * mu1_mu2 + c1) * (2.0 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = numerator / denominator
    return round(float(ssim_map.mean().item()), 4)


def load_rgb_array(path: Path, resize_to: Optional[Tuple[int, int]] = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if resize_to is not None and img.size != resize_to:
        img = img.resize(resize_to, Image.Resampling.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def evaluate_directories(pred_dir: Path, gt_dir: Path) -> Dict[str, Any]:
    """Match predicted images (including '<stem>_enhanced' outputs) against ground-truth files."""
    gt_by_stem = {
        p.stem: p for p in sorted(gt_dir.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTS
    }
    pred_files = [
        p for p in sorted(pred_dir.iterdir()) if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]

    records: List[Dict[str, Any]] = []
    for pred_path in pred_files:
        stem = pred_path.stem
        lookup_stem = stem[:-9] if stem.endswith("_enhanced") else stem
        gt_path = gt_by_stem.get(lookup_stem)
        if gt_path is None:
            continue

        gt_img = Image.open(gt_path).convert("RGB")
        gt_arr = np.asarray(gt_img, dtype=np.float32) / 255.0
        pred_arr = load_rgb_array(pred_path, resize_to=gt_img.size)

        psnr_val = compute_psnr(pred_arr, gt_arr)
        ssim_val = compute_ssim(pred_arr, gt_arr)
        records.append({
            "image": lookup_stem,
            "pred_file": pred_path.name,
            "gt_file": gt_path.name,
            "psnr_db": psnr_val,
            "ssim": ssim_val,
        })

    if not records:
        raise FileNotFoundError(
            f"No matching image pairs found between '{pred_dir}' and '{gt_dir}'."
        )

    finite_psnrs = [r["psnr_db"] for r in records if math.isfinite(r["psnr_db"])]
    mean_psnr = round(sum(finite_psnrs) / len(finite_psnrs), 4) if finite_psnrs else float("inf")
    mean_ssim = round(sum(r["ssim"] for r in records) / len(records), 4)

    return {
        "evaluated_pairs": len(records),
        "mean_psnr_db": mean_psnr,
        "mean_ssim": mean_ssim,
        "per_image": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PSNR and SSIM for Dual-Attention-GAN-LLIE outputs."
    )
    parser.add_argument("--pred-dir", type=str, required=True, help="Directory of enhanced images")
    parser.add_argument("--gt-dir", type=str, required=True, help="Directory of ground-truth images")
    parser.add_argument("--json-out", type=str, default=None, help="Optional JSON report path")
    args = parser.parse_args()

    summary = evaluate_directories(Path(args.pred_dir), Path(args.gt_dir))
    print(
        f"[+] Evaluated {summary['evaluated_pairs']} image pairs | "
        f"Mean PSNR: {summary['mean_psnr_db']} dB | Mean SSIM: {summary['mean_ssim']}"
    )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[+] Saved evaluation report to: {out_path}")


if __name__ == "__main__":
    main()