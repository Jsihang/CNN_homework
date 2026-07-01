import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from src.datasets import CIFAR10_CLASSES
from src.models import ConditionalGenerator, sample_noise
from src.resnet_gan import ResNetConditionalGenerator
from src.utils import denormalize, resolve_device


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a labeled CIFAR-10 class grid from a conditional GAN.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/conditional_gan/labeled_grid.png")
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--no-ema", action="store_true", help="Use raw generator weights even if EMA weights exist.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def tensor_to_image(tensor):
    tensor = denormalize(tensor).mul(255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(tensor)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint.get("args", {})
    latent_dim = int(cfg.get("latent_dim", 128))
    g_channels = int(cfg.get("g_channels", 64))

    arch = cfg.get("arch", "cnn")
    if arch == "resnet":
        generator = ResNetConditionalGenerator(latent_dim=latent_dim, base_channels=g_channels).to(device)
    else:
        generator = ConditionalGenerator(latent_dim=latent_dim, base_channels=g_channels).to(device)
    state_key = "generator" if args.no_ema or "ema_generator" not in checkpoint else "ema_generator"
    generator.load_state_dict(checkpoint[state_key])
    generator.eval()

    labels = torch.arange(10, device=device).repeat_interleave(args.samples_per_class)
    z = sample_noise(labels.numel(), latent_dim, device)
    with torch.no_grad():
        images = generator(z, labels).cpu()

    cell = 32
    label_w = 96
    rows = 10
    cols = args.samples_per_class
    canvas = Image.new("RGB", (label_w + cols * cell, rows * cell), "white")
    draw = ImageDraw.Draw(canvas)
    for row, name in enumerate(CIFAR10_CLASSES):
        y = row * cell
        draw.text((6, y + 10), name, fill=(0, 0, 0))
        for col in range(cols):
            idx = row * cols + col
            canvas.paste(tensor_to_image(images[idx]), (label_w + col * cell, y))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"Saved labeled conditional grid to {out}")


if __name__ == "__main__":
    main()
