import argparse
import time
from pathlib import Path

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.datasets import LocalCIFAR10
from src.diffaugment import diff_augment
from src.ema import EMA
from src.models import ConditionalGenerator, ProjectionDiscriminator, sample_noise, weights_init
from src.resnet_gan import ResNetConditionalGenerator, ResNetProjectionDiscriminator, init_resnet_weights
from src.utils import append_history, resolve_device, save_checkpoint, save_loss_plot, save_sample_grid, seed_everything, write_json


DEFAULT_DATA_DIR = "/data1/nHome1/xieqihu/dataset/CV/CIFAR10_data/cifar-10-batches-py"


def parse_args():
    parser = argparse.ArgumentParser(description="Train a class-conditional GAN on local CIFAR-10.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", default="outputs/conditional_gan")
    parser.add_argument("--arch", choices=["cnn", "resnet"], default="cnn")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--g-channels", type=int, default=64)
    parser.add_argument("--d-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--diffaugment", action="store_true")
    parser.add_argument("--diffaugment-policy", default="color,translation,cutout")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_loader(args):
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    max_items = 1024 if args.fast_dev_run and args.max_train_items is None else args.max_train_items
    dataset = LocalCIFAR10(args.data_dir, train=True, transform=transform, max_items=max_items)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


def make_condition_grid(generator, latent_dim, device, samples_per_class=8):
    labels = torch.arange(10, device=device).repeat_interleave(samples_per_class)
    z = sample_noise(labels.numel(), latent_dim, device)
    generator.eval()
    with torch.no_grad():
        images = generator(z, labels).cpu()
    generator.train()
    return images


def build_models(args, device):
    if args.arch == "resnet":
        generator = ResNetConditionalGenerator(args.latent_dim, base_channels=args.g_channels)
        discriminator = ResNetProjectionDiscriminator(base_channels=args.d_channels)
        generator.apply(init_resnet_weights)
    else:
        generator = ConditionalGenerator(args.latent_dim, base_channels=args.g_channels)
        discriminator = ProjectionDiscriminator(base_channels=args.d_channels)
        generator.apply(weights_init)
    return generator.to(device), discriminator.to(device)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(vars(args), out_dir / "config.json")

    loader = build_loader(args)
    generator, discriminator = build_models(args, device)
    ema = EMA(generator, decay=args.ema_decay) if args.ema else None

    g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        d_losses = []
        g_losses = []
        progress = tqdm(loader, desc=f"Conditional epoch {epoch}/{args.epochs}", leave=False)

        for real_images, labels in progress:
            real_images = real_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = real_images.size(0)

            z = sample_noise(batch_size, args.latent_dim, device)
            fake_images = generator(z, labels).detach()
            d_real_images = diff_augment(real_images, args.diffaugment_policy) if args.diffaugment else real_images
            d_fake_images = diff_augment(fake_images, args.diffaugment_policy) if args.diffaugment else fake_images
            real_score = discriminator(d_real_images, labels)
            fake_score = discriminator(d_fake_images, labels)
            d_loss = F.relu(1.0 - real_score).mean() + F.relu(1.0 + fake_score).mean()
            discriminator.zero_grad(set_to_none=True)
            d_loss.backward()
            d_optimizer.step()

            z = sample_noise(batch_size, args.latent_dim, device)
            fake_images = generator(z, labels)
            g_fake_images = diff_augment(fake_images, args.diffaugment_policy) if args.diffaugment else fake_images
            g_loss = -discriminator(g_fake_images, labels).mean()
            generator.zero_grad(set_to_none=True)
            g_loss.backward()
            g_optimizer.step()
            if ema is not None:
                ema.update(generator)

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())
            progress.set_postfix(d_loss=f"{d_loss.item():.3f}", g_loss=f"{g_loss.item():.3f}")

        mean_d = sum(d_losses) / len(d_losses)
        mean_g = sum(g_losses) / len(g_losses)
        append_history(
            out_dir / "history.csv",
            {
                "epoch": epoch,
                "d_loss": f"{mean_d:.6f}",
                "g_loss": f"{mean_g:.6f}",
                "seconds": f"{time.time() - start:.2f}",
                "gan_type": f"conditional-{args.arch}-hinge",
            },
        )
        print(f"Epoch {epoch:03d}: d_loss={mean_d:.4f}, g_loss={mean_g:.4f}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            sample_generator = ema.ema_model if ema is not None else generator
            images = make_condition_grid(sample_generator, args.latent_dim, device)
            save_sample_grid(images, out_dir / "samples" / f"epoch_{epoch:03d}.png", nrow=8)

        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            extra = {"ema_generator": ema.state_dict()} if ema is not None else None
            save_checkpoint(
                out_dir / "checkpoints" / f"epoch_{epoch:03d}.pt",
                generator,
                discriminator,
                g_optimizer,
                d_optimizer,
                args,
                epoch,
                extra=extra,
            )

        if args.fast_dev_run:
            break

    extra = {"ema_generator": ema.state_dict()} if ema is not None else None
    save_checkpoint(out_dir / "checkpoints" / "latest.pt", generator, discriminator, g_optimizer, d_optimizer, args, epoch, extra=extra)
    save_loss_plot(out_dir / "history.csv", out_dir / "loss_curve.png")
    print(f"Done. Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
