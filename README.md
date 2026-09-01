# Dual-Attention GAN: A Robust Framework for Low-Light Image Enhancement using Efficient Residual Blocks

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Conference](https://img.shields.io/badge/EFAST-2026-blue.svg)](https://efast.pust.ac.bd)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Official implementation and model architecture for the undergraduate research thesis conducted at **East Delta University** and presented at **EFAST 2026**[cite: 1, 2, 5].

---

## 👥 Authors & Academic Affiliation

* **Author:** Irtaz Irfan (Student ID: 221008612)
* **Supervisor:** Golam Moktader Daiyan, Assistant Professor
* **Department:** Computer Science & Engineering, School of Science, Engineering & Technology[cite: 1, 5]
* **Institution:** East Delta University, Chittagong, Bangladesh[cite: 1, 5]
* **Defense Date:** March 2026 (Grade Awarded: A / 4.00 CGPA)[cite: 1, 5]

---

## 📖 Abstract

Low-light image enhancement is critical for downstream computer vision tasks like autonomous driving and surveillance, which suffer under severe visibility degradation, intensive noise, and color distortion[cite: 1, 5]. To overcome the generalization limits of traditional priors and the heavy overhead of deep learning models, this thesis proposes an advanced Generative Adversarial Network (GAN) optimized for high-fidelity restoration[cite: 1, 5]. 

The framework features a U-Net generator integrated with nine sequential Efficient Residual Blocks for multi-scale feature extraction without spatial resolution loss[cite: 1, 5]. It introduces a dual-attention mechanism combining Enhanced Channel-Spatial and Squeeze-and-Excitation (SE) modules to dynamically recalibrate spatial and channel feature dependencies, balancing illumination adjustment with noise suppression[cite: 1, 5]. Adversarially, a Spectrally Normalized Patch-GAN discriminator enforces localized structural constraints to preserve sharp, high-frequency textures[cite: 1, 5]. Validated on the LoLI-Street dataset, the model achieves a Peak Signal-to-Noise Ratio (PSNR) of 32.21 dB, a Structural Similarity Index (SSIM) of 0.9678, and a Learned Perceptual Image Patch Similarity (LPIPS) score of 0.0172, significantly outperforming state-of-the-art benchmarks including Retinexformer and CUE[cite: 1, 5].

---

## 🏛️ System Architecture

### 1. U-Net Generator with Dual-Attention Bottleneck
The generator consists of:
* **Encoder:** 4 downsampling blocks utilizing $3 \times 3$ stride-2 convolutions with Instance Normalization and ReLU activations[cite: 1, 5].
* **Bottleneck:** 9 sequential **Efficient Residual Blocks** built with Depthwise Separable Convolutions to decouple spatial filtering from channel mixing[cite: 1, 5].
* **Dual-Attention Mechanism:**
  * *Channel Attention:* Global Average Pooling & Global Max Pooling $\to$ Shared MLP ($r=16$) $\to$ Sigmoid[cite: 1, 5].
  * *Spatial Attention:* Channel-wise pooling $\to 7 \times 7$ Convolution $\to$ Sigmoid[cite: 1, 5].
  * *Fusion & Refinement:* Combined via element-wise addition and passed through a Squeeze-and-Excitation (SE) module[cite: 1, 5].
* **Decoder:** 4 upsampling stages using Nearest-Neighbor interpolation, skip-connection concatenations from encoder stages, and a $7 \times 7$ Convolution mapped via Tanh to $[-1, 1]$[cite: 1, 5].

### 2. Spectrally Normalized PatchGAN Discriminator
* 5-Layer Fully Convolutional Network classifying $N \times N$ local image patches[cite: 1, 5].
* Implements **Spectral Normalization** across all layers to bound the Lipschitz constant, preventing gradient explosion and stabilizing adversarial minimax training[cite: 1, 5].

---

## 📐 Loss Formulation & Optimization

The network optimizes a multi-objective loss function balancing adversarial realism and pixel-level structural fidelity[cite: 1, 5]:

$$\mathcal{L}_{Total} = \mathcal{L}_{GAN}(G, D) + \lambda \mathcal{L}_1(G)$$

$$\mathcal{L}_{GAN}(G, D) = \mathbb{E}_{y}[\log D(y)] + \mathbb{E}_{x}[\log(1 - D(G(x)))]$$

$$\mathcal{L}_1(G) = \mathbb{E}_{x, y}[\Vert{}y - G(x)\Vert{}_1]$$

* **Reconstruction Weight:** $\lambda = 100$[cite: 1, 5]
* **Optimizer:** Adam ($\beta_1 = 0.5, \beta_2 = 0.999$)[cite: 1, 5]
* **Learning Rates:** Generator = $2 \times 10^{-4}$, Discriminator = $1 \times 10^{-4}$[cite: 1, 5]
* **Batch Size:** 32 (with gradient accumulation)[cite: 1, 5]
* **Epochs:** 50 epochs on the LoLI-Street Dataset[cite: 1, 5]

---

## 📊 Benchmark Results on LoLI-Street Dataset

Evaluated on the 500-image dense low-light split of the **LoLI-Street Benchmark Dataset** (5,000 paired training subset)[cite: 1, 5]:

| Method | Venue / Year | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| :--- | :---: | :---: | :---: | :---: |
| Retinexformer[cite: 1, 5] | ICCV 2023 | 27.87 | 0.3398 | 0.3037 |
| SCI[cite: 1, 5] | CVPR 2022 | 27.79 | 0.3394 | 0.3048 |
| PairLIE[cite: 1, 5] | CVPR 2023 | 27.66 | 0.8702 | 0.0357 |
| FourLLIE[cite: 1, 5] | ACM MM 2023 | 28.07 | 0.8828 | 0.1188 |
| RQ-LLIE[cite: 1, 5] | ICCV 2023 | 29.03 | 0.9167 | 0.0326 |
| CUE[cite: 1, 5] | ICCV 2023 | 30.58 | 0.9100 | 0.0201 |
| Diff-LL[cite: 1, 5] | ACM TOG 2023 | 31.04 | 0.9165 | 0.0273 |
| TriFuse[cite: 1, 5] | SOTA 2024 | 31.67 | 0.9214 | 0.0201 |
| LL-Former[cite: 1, 5] | AAAI 2023 | 31.62 | 0.9274 | **0.0131** |
| **Proposed Dual-Attention GAN (Ours)**[cite: 1, 5] | **Thesis 2026** | **32.21** | **0.9678** | 0.0172 |

---

## 🚀 Getting Started

### 1. Installation
```bash
git clone [https://github.com/irtazirfan08-source/Dual-Attention-GAN-LLIE.git](https://github.com/irtazirfan08-source/Dual-Attention-GAN-LLIE.git)
cd Dual-Attention-GAN-LLIE
pip install -r requirements.txt