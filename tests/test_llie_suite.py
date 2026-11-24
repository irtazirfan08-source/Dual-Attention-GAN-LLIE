"""Non-invasive unit and integration tests for Dual-Attention-GAN-LLIE."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from eval_metrics import compute_psnr, compute_ssim, evaluate_directories
from infer import denormalize, get_transform, load_generator, run_inference
from model import (
    DepthwiseSeparableConv,
    DualAttentionGenerator,
    EfficientResidualBlock,
    EnhancedChannelSpatialAttention,
    PatchGANDiscriminator,
    SqueezeAndExcitation,
)


@pytest.mark.parametrize("resolution", [128, 256])
def test_generator_forward_shapes_and_tanh_bounds(resolution: int):
    gen = DualAttentionGenerator(in_channels=3, out_channels=3, num_residuals=2)
    gen.eval()
    x = torch.randn(1, 3, resolution, resolution)
    with torch.no_grad():
        out = gen(x)

    assert out.shape == (1, 3, resolution, resolution)
    assert float(out.min()) >= -1.0
    assert float(out.max()) <= 1.0


def test_patchgan_discriminator_validity_map_shape():
    disc = PatchGANDiscriminator(in_channels=3)
    disc.eval()
    x = torch.randn(2, 3, 256, 256)
    with torch.no_grad():
        validity = disc(x)

    assert validity.shape == (2, 1, 30, 30)


def test_attention_and_residual_submodules():
    x = torch.randn(2, 64, 32, 32)
    assert DepthwiseSeparableConv(64, 64)(x).shape == (2, 64, 32, 32)
    assert EnhancedChannelSpatialAttention(64)(x).shape == (2, 64, 32, 32)
    assert SqueezeAndExcitation(64)(x).shape == (2, 64, 32, 32)
    assert EfficientResidualBlock(64)(x).shape == (2, 64, 32, 32)


def test_inference_pipeline_generates_enhanced_and_comparison_images(tmp_path: Path):
    input_path = tmp_path / "low_light_sample.png"
    Image.fromarray(np.full((64, 80, 3), 30, dtype=np.uint8)).save(input_path)

    device = torch.device("cpu")
    model = load_generator(weights_path=None, device=device)
    out_dir = tmp_path / "outputs"

    run_inference(
        image_path=str(input_path),
        model=model,
        device=device,
        output_dir=str(out_dir),
        save_comparison=True,
    )

    enhanced_path = out_dir / "low_light_sample_enhanced.png"
    comparison_path = out_dir / "low_light_sample_comparison.png"

    assert enhanced_path.is_file()
    assert comparison_path.is_file()

    with Image.open(enhanced_path) as enh_img:
        assert enh_img.size == (80, 64)
    with Image.open(comparison_path) as comp_img:
        assert comp_img.size == (160, 64)


def test_transform_and_denormalize_roundtrip():
    img = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
    tensor = get_transform(img_size=64)(img).unsqueeze(0)
    assert tensor.shape == (1, 3, 64, 64)
    restored = denormalize(tensor)
    assert restored.size == (64, 64)


def test_psnr_and_ssim_metric_invariants():
    rng = np.random.default_rng(42)
    clean = rng.uniform(0.1, 0.9, size=(64, 64, 3)).astype(np.float32)

    assert math.isinf(compute_psnr(clean, clean))
    assert compute_ssim(clean, clean) == pytest.approx(1.0, abs=1e-4)

    noisy = np.clip(clean + 0.05, 0.0, 1.0)
    psnr_noisy = compute_psnr(noisy, clean)
    ssim_noisy = compute_ssim(noisy, clean)

    assert 20.0 < psnr_noisy < 35.0
    assert 0.5 < ssim_noisy < 1.0


def test_evaluate_directories_matches_enhanced_suffix(tmp_path: Path):
    pred_dir = tmp_path / "preds"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()

    gt_arr = np.full((64, 64, 3), 180, dtype=np.uint8)
    pred_arr = np.full((64, 64, 3), 170, dtype=np.uint8)

    Image.fromarray(gt_arr).save(gt_dir / "street_01.png")
    Image.fromarray(pred_arr).save(pred_dir / "street_01_enhanced.png")

    summary = evaluate_directories(pred_dir, gt_dir)
    assert summary["evaluated_pairs"] == 1
    assert summary["mean_psnr_db"] > 25.0
    assert summary["mean_ssim"] > 0.95
    assert summary["per_image"][0]["image"] == "street_01"