import argparse
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path("outputs/torch_cache").resolve()))

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from evaluate import build_generator_from_checkpoint, inception_forward, load_inception
from src.datasets import CIFAR10_CLASSES, LocalCIFAR10
from src.models import sample_noise
from src.utils import resolve_device


DEFAULT_DATA_DIR = "/data1/nHome1/xieqihu/dataset/CV/CIFAR10_data/cifar-10-batches-py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a class-wise showcase grid from the best unconditional GAN."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", default="outputs/class_grid.png")
    parser.add_argument("--num-candidates", type=int, default=3000)
    parser.add_argument("--real-per-class", type=int, default=1000)
    parser.add_argument("--images-per-class", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument("--model-type", choices=["auto", "unconditional", "conditional"], default="auto")
    parser.add_argument("--arch", choices=["auto", "cnn", "resnet"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-ema", action="store_true")
    return parser.parse_args()


def tensor_to_pil(image):
    image = image.detach().cpu().mul(255).clamp(0, 255).to(torch.uint8)
    image = image.permute(1, 2, 0).numpy()
    return Image.fromarray(image)


def get_font(size):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def collect_class_centroids(model, args, device):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = LocalCIFAR10(args.data_dir, train=True, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    sums = torch.zeros(10, args.feature_dim)
    counts = torch.zeros(10)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Real class centroids", leave=False):
            images = images.to(device)
            labels = labels.long()
            features, _ = inception_forward(model, images, args.feature_dim)
            features = F.normalize(features.cpu(), dim=1)
            for class_id in range(10):
                mask = labels == class_id
                if mask.any() and counts[class_id] < args.real_per_class:
                    need = int(args.real_per_class - counts[class_id].item())
                    selected = features[mask][:need]
                    sums[class_id] += selected.sum(dim=0)
                    counts[class_id] += selected.size(0)
            if torch.all(counts >= args.real_per_class):
                break

    centroids = F.normalize(sums / counts.clamp_min(1).unsqueeze(1), dim=1)
    return centroids.to(device)


def collect_ranked_fakes(model, generator, latent_dim, centroids, args, device, conditional):
    ranked = defaultdict(list)
    generated = 0
    generator.eval()
    with torch.no_grad():
        progress = tqdm(total=args.num_candidates, desc="Generated candidates", leave=False)
        while generated < args.num_candidates:
            batch_size = min(args.batch_size, args.num_candidates - generated)
            z = sample_noise(batch_size, latent_dim, device)
            if conditional:
                labels = torch.arange(batch_size, device=device) % 10
                images = generator(z, labels).mul(0.5).add(0.5).clamp(0, 1)
            else:
                images = generator(z).mul(0.5).add(0.5).clamp(0, 1)
            features, _ = inception_forward(model, images, args.feature_dim)
            features = F.normalize(features, dim=1)
            scores = features @ centroids.T
            best_scores, pred_labels = scores.max(dim=1)
            for image, label, score in zip(images.cpu(), pred_labels.cpu(), best_scores.cpu()):
                ranked[int(label)].append((float(score), image))
            generated += batch_size
            progress.update(batch_size)
        progress.close()

    for class_id in ranked:
        ranked[class_id].sort(key=lambda item: item[0], reverse=True)
    return ranked


def make_grid(ranked, out, images_per_class):
    tile = 32
    scale = 3
    label_width = 120
    header_height = 32
    gap = 6
    row_height = tile * scale + gap
    width = label_width + images_per_class * tile * scale + (images_per_class - 1) * gap
    height = header_height + 10 * row_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = get_font(18)

    draw.text((label_width, 6), "Best GAN samples grouped by CIFAR-10 class", fill=(20, 20, 20), font=font)
    for class_id, name in enumerate(CIFAR10_CLASSES):
        y = header_height + class_id * row_height
        draw.text((8, y + 34), name, fill=(20, 20, 20), font=font)
        samples = ranked.get(class_id, [])[:images_per_class]
        for col, (_, image) in enumerate(samples):
            x = label_width + col * (tile * scale + gap)
            pil = tensor_to_pil(image).resize((tile * scale, tile * scale), Image.Resampling.NEAREST)
            canvas.paste(pil, (x, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator, latent_dim, conditional, state_key, arch = build_generator_from_checkpoint(checkpoint, args, device)
    inception, status = load_inception(device, allow_download=False, untrained_ok=False)
    centroids = collect_class_centroids(inception, args, device)
    ranked = collect_ranked_fakes(inception, generator, latent_dim, centroids, args, device, conditional)
    out = Path(args.out)
    make_grid(ranked, out, args.images_per_class)
    print(f"Saved class-wise grid to {out}")
    print(f"Generator state: {state_key}; arch: {arch}; Inception: {status}")
    for class_id, name in enumerate(CIFAR10_CLASSES):
        print(f"{name}: {len(ranked.get(class_id, []))} candidates")


if __name__ == "__main__":
    main()
