import argparse
import os
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from model import DualAttentionGenerator


def get_transform(img_size: int = 256):
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def denormalize(tensor: torch.Tensor) -> Image.Image:
    # Scale from [-1, 1] back to [0, 1]
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    to_pil = T.ToPILImage()
    return to_pil(tensor.squeeze(0).cpu())


def load_generator(weights_path: str = None, device: torch.device = None) -> torch.nn.Module:
    model = DualAttentionGenerator(in_channels=3, out_channels=3, num_residuals=9)
    if weights_path and os.path.isfile(weights_path):
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"[+] Loaded weights from: {weights_path}")
    else:
        print("[!] Note: No checkpoint path provided. Running with initialized architecture.")
    model.to(device)
    model.eval()
    return model


def run_inference(image_path: str, model: torch.nn.Module, device: torch.device, output_dir: str, save_comparison: bool = True):
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size

    transform = get_transform()
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        enhanced_tensor = model(input_tensor)

    enhanced_img = denormalize(enhanced_tensor).resize(orig_size, Image.Resampling.BICUBIC)

    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem

    out_path = os.path.join(output_dir, f"{base_name}_enhanced.png")
    enhanced_img.save(out_path)
    print(f"[+] Saved enhanced output: {out_path}")

    if save_comparison:
        comp_w = orig_size[0] * 2
        comp_h = orig_size[1]
        comparison = Image.new("RGB", (comp_w, comp_h))
        comparison.paste(img, (0, 0))
        comparison.paste(enhanced_img, (orig_size[0], 0))
        comp_path = os.path.join(output_dir, f"{base_name}_comparison.png")
        comparison.save(comp_path)
        print(f"[+] Saved comparison: {comp_path}")


def main():
    parser = argparse.ArgumentParser(description="Zero-Config Inference for Dual-Attention-GAN-LLIE")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image or folder")
    parser.add_argument("--weights", "-w", type=str, default=None, help="Path to generator checkpoint (.pth)")
    parser.add_argument("--output", "-o", type=str, default="outputs", help="Output directory path")
    parser.add_argument("--device", "-d", type=str, default=None, help="Device ('cuda' or 'cpu')")
    parser.add_argument("--no-comparison", action="store_true", help="Skip saving side-by-side comparison")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Inference device: {device}")

    model = load_generator(args.weights, device)

    input_path = Path(args.input)
    if input_path.is_file():
        run_inference(str(input_path), model, device, args.output, not args.no_comparison)
    elif input_path.is_dir():
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = [p for p in input_path.iterdir() if p.suffix.lower() in valid_exts]
        if not images:
            print(f"[!] No valid image files found in {input_path}")
            return
        for img_file in images:
            run_inference(str(img_file), model, device, args.output, not args.no_comparison)
    else:
        print(f"[!] Path not found: {args.input}")


if __name__ == "__main__":
    main()