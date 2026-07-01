import csv
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def denormalize(images):
    return images.mul(0.5).add(0.5).clamp(0, 1)


def save_sample_grid(images, path, nrow=8):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(images), path, nrow=nrow)


def save_real_grid(dataloader, path, nrow=8, max_images=64):
    images, _ = next(iter(dataloader))
    save_sample_grid(images[:max_images], path, nrow=nrow)


def write_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)


def append_history(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_loss_plot(history_csv, output_path):
    cache_dir = Path("/tmp/matplotlib-codex")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib.pyplot as plt

    rows = []
    with Path(history_csv).open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows.extend(reader)
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    d_loss = [float(row["d_loss"]) for row in rows]
    g_loss = [float(row["g_loss"]) for row in rows]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, d_loss, label="Discriminator")
    plt.plot(epochs, g_loss, label="Generator")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


def save_checkpoint(path, generator, discriminator, g_optimizer, d_optimizer, args, epoch, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict() if discriminator is not None else None,
        "g_optimizer": g_optimizer.state_dict() if g_optimizer is not None else None,
        "d_optimizer": d_optimizer.state_dict() if d_optimizer is not None else None,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)
