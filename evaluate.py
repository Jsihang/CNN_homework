import argparse
import math
import os
from pathlib import Path

os.environ.setdefault("TORCH_HOME", str(Path("outputs/torch_cache").resolve()))

import numpy as np
import torch
from torch import nn
from scipy import linalg
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from tqdm import tqdm

from src.datasets import LocalCIFAR10
from src.models import ConditionalGenerator, Generator, sample_noise
from src.resnet_gan import ResNetConditionalGenerator
from src.utils import resolve_device, write_json


DEFAULT_DATA_DIR = "/data1/nHome1/xieqihu/dataset/CV/CIFAR10_data/cifar-10-batches-py"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a CIFAR-10 GAN checkpoint with IS and FID.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", default="outputs/metrics.json")
    parser.add_argument("--model-type", choices=["auto", "unconditional", "conditional"], default="auto")
    parser.add_argument("--arch", choices=["auto", "cnn", "resnet"], default="auto")
    parser.add_argument("--no-ema", action="store_true", help="Use raw generator weights if EMA weights exist.")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow torchvision to download Inception weights if they are not cached.",
    )
    parser.add_argument(
        "--untrained-inception-ok",
        action="store_true",
        help="Fall back to random Inception weights. This is only for pipeline debugging.",
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


class InceptionEvaluator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._features = None
        self.model.avgpool.register_forward_hook(self._capture_avgpool)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def _capture_avgpool(self, module, inputs, output):
        self._features = torch.flatten(output, start_dim=1)

    def forward(self, images):
        resized = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        normalized = (resized - self.mean) / self.std
        logits = self.model(normalized)
        if isinstance(logits, tuple):
            logits = logits[0]
        return self._features, logits


def load_inception(device, allow_download=False, untrained_ok=False):
    if untrained_ok and not allow_download:
        model = inception_v3(weights=None, transform_input=False, aux_logits=False, init_weights=False)
        model.eval()
        return InceptionEvaluator(model).to(device), "untrained_debug_only"

    try:
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, transform_input=False, aux_logits=True)
        status = "pretrained"
    except Exception as exc:
        if not allow_download and not untrained_ok:
            raise RuntimeError(
                "Could not load cached Inception weights. Re-run with --allow-download on a networked "
                "machine, or use --untrained-inception-ok only to verify the code path."
            ) from exc
        if not untrained_ok:
            weights = Inception_V3_Weights.DEFAULT
            model = inception_v3(weights=weights, transform_input=False, aux_logits=True)
            status = "pretrained"
        else:
            model = inception_v3(weights=None, transform_input=False, aux_logits=False)
            status = "untrained_debug_only"

    model.eval()
    return InceptionEvaluator(model).to(device), status


def inception_forward(model, images, feature_dim=None):
    features, logits = model(images)
    if feature_dim is not None:
        features = features[:, :feature_dim]
    return features, logits


def collect_real_features(model, args, device):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = LocalCIFAR10(args.data_dir, train=True, transform=transform, max_items=args.num_samples)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    feats = []
    logits = []
    seen = 0
    for images, _ in tqdm(loader, desc="Real features", leave=False):
        images = images.to(device)
        with torch.no_grad():
            features, batch_logits = inception_forward(model, images, args.feature_dim)
        feats.append(features.cpu())
        logits.append(batch_logits.cpu())
        seen += images.size(0)
        if seen >= args.num_samples:
            break
    return torch.cat(feats, dim=0)[: args.num_samples], torch.cat(logits, dim=0)[: args.num_samples]


def collect_fake_features(model, generator, latent_dim, args, device, conditional=False):
    feats = []
    logits = []
    generated = 0
    with torch.no_grad():
        progress = tqdm(total=args.num_samples, desc="Fake features", leave=False)
        while generated < args.num_samples:
            batch_size = min(args.batch_size, args.num_samples - generated)
            z = sample_noise(batch_size, latent_dim, device)
            if conditional:
                labels = torch.randint(0, 10, (batch_size,), device=device)
                images = generator(z, labels).mul(0.5).add(0.5).clamp(0, 1)
            else:
                images = generator(z).mul(0.5).add(0.5).clamp(0, 1)
            features, batch_logits = inception_forward(model, images, args.feature_dim)
            feats.append(features.cpu())
            logits.append(batch_logits.cpu())
            generated += batch_size
            progress.update(batch_size)
        progress.close()
    return torch.cat(feats, dim=0), torch.cat(logits, dim=0)


def build_generator_from_checkpoint(checkpoint, args, device):
    cfg = checkpoint.get("args", {})
    latent_dim = int(cfg.get("latent_dim", 128))
    g_channels = int(cfg.get("g_channels", 64))

    model_type = args.model_type
    if model_type == "auto":
        if "gan_type" in cfg:
            model_type = "unconditional"
        else:
            model_type = "conditional" if cfg.get("arch") in {"cnn", "resnet"} or "ema_generator" in checkpoint else "unconditional"

    arch = args.arch
    if arch == "auto":
        arch = cfg.get("arch", "cnn")

    if model_type == "conditional":
        if arch == "resnet":
            generator = ResNetConditionalGenerator(latent_dim=latent_dim, base_channels=g_channels).to(device)
        else:
            generator = ConditionalGenerator(latent_dim=latent_dim, base_channels=g_channels).to(device)
        state_key = "generator" if args.no_ema or "ema_generator" not in checkpoint else "ema_generator"
        generator.load_state_dict(checkpoint[state_key])
        return generator.eval(), latent_dim, True, state_key, arch

    generator = Generator(latent_dim=latent_dim, base_channels=g_channels).to(device)
    state_key = "generator" if args.no_ema or "ema_generator" not in checkpoint else "ema_generator"
    generator.load_state_dict(checkpoint[state_key])
    return generator.eval(), latent_dim, False, state_key, "dcgan"


def calculate_fid(real_features, fake_features):
    real = real_features.numpy().astype(np.float64)
    fake = fake_features.numpy().astype(np.float64)
    mu_real, mu_fake = real.mean(axis=0), fake.mean(axis=0)
    sigma_real, sigma_fake = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma_real + offset) @ (sigma_fake + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean))


def calculate_inception_score(logits, splits=10):
    probs = torch.softmax(logits, dim=1).numpy()
    scores = []
    split_size = max(1, probs.shape[0] // splits)
    for start in range(0, probs.shape[0], split_size):
        part = probs[start : start + split_size]
        if len(part) == 0:
            continue
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + 1e-12) - np.log(py + 1e-12))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return float(np.mean(scores)), float(np.std(scores))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator, latent_dim, conditional, state_key, arch = build_generator_from_checkpoint(checkpoint, args, device)

    inception, inception_status = load_inception(
        device, allow_download=args.allow_download, untrained_ok=args.untrained_inception_ok
    )
    real_features, _ = collect_real_features(inception, args, device)
    fake_features, fake_logits = collect_fake_features(inception, generator, latent_dim, args, device, conditional=conditional)
    fid = calculate_fid(real_features, fake_features)
    is_mean, is_std = calculate_inception_score(fake_logits)

    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "num_samples": args.num_samples,
        "feature_dim": args.feature_dim,
        "model_type": "conditional" if conditional else "unconditional",
        "arch": arch,
        "generator_state": state_key,
        "inception_status": inception_status,
        "fid": fid,
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
        "note": "Use pretrained Inception for report numbers. untrained_debug_only is not a valid final metric.",
    }
    write_json(result, args.out)
    print(result)


if __name__ == "__main__":
    main()
