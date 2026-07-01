import argparse
from pathlib import Path

import torch

from src.models import Generator, sample_noise
from src.utils import resolve_device, save_sample_grid


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images from a trained GAN checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="outputs/generated_grid.png")
    parser.add_argument("--num-images", type=int, default=64)
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--no-ema", action="store_true", help="Use raw generator weights if EMA weights exist.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    cfg = checkpoint.get("args", {})

    latent_dim = int(cfg.get("latent_dim", 128))
    g_channels = int(cfg.get("g_channels", 64))
    generator = Generator(latent_dim=latent_dim, base_channels=g_channels).to(device)
    state_key = "generator" if args.no_ema or "ema_generator" not in checkpoint else "ema_generator"
    generator.load_state_dict(checkpoint[state_key])
    generator.eval()

    images = []
    remaining = args.num_images
    with torch.no_grad():
        while remaining > 0:
            batch_size = min(remaining, 128)
            z = sample_noise(batch_size, latent_dim, device)
            images.append(generator(z).cpu())
            remaining -= batch_size
    images = torch.cat(images, dim=0)

    out = Path(args.out)
    save_sample_grid(images, out, nrow=args.nrow)
    print(f"Saved {args.num_images} generated images to {out}")


if __name__ == "__main__":
    main()
