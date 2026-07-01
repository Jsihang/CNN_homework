import argparse
import time
from pathlib import Path

import torch
from torch import autograd, nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.datasets import LocalCIFAR10
from src.diffaugment import diff_augment
from src.ema import EMA
from src.models import make_models, sample_noise
from src.utils import (
    append_history,
    resolve_device,
    save_checkpoint,
    save_loss_plot,
    save_real_grid,
    save_sample_grid,
    seed_everything,
    write_json,
)


DEFAULT_DATA_DIR = "/data1/nHome1/xieqihu/dataset/CV/CIFAR10_data/cifar-10-batches-py"


def parse_args():
    parser = argparse.ArgumentParser(description="Train DCGAN or WGAN-GP on local CIFAR-10.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to cifar-10-batches-py.")
    parser.add_argument("--out-dir", default="outputs/dcgan", help="Directory for checkpoints and samples.")
    parser.add_argument("--gan-type", choices=["dcgan", "wgan-gp"], default="dcgan")
    parser.add_argument("--loss", choices=["bce", "lsgan"], default="bce", help="Loss for --gan-type dcgan.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--g-channels", type=int, default=64)
    parser.add_argument("--d-channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--n-critic", type=int, default=5, help="Discriminator updates per generator update for WGAN-GP.")
    parser.add_argument("--gp-lambda", type=float, default=10.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--diffaugment", action="store_true")
    parser.add_argument("--diffaugment-policy", default="color,translation,cutout")
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--max-train-items", type=int, default=None, help="Use a subset for quick debugging.")
    parser.add_argument("--fast-dev-run", action="store_true", help="Run only a few batches to verify the pipeline.")
    parser.add_argument("--device", default="auto", help="auto, cuda, or cpu.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataloader(args):
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


def gradient_penalty(discriminator, real_images, fake_images, device):
    batch_size = real_images.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real_images + (1 - alpha) * fake_images
    interpolates.requires_grad_(True)
    scores = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=scores,
        inputs=interpolates,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(batch_size, -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()


def train_dcgan_step(real_images, generator, discriminator, g_optimizer, d_optimizer, criterion, args, device, ema=None):
    batch_size = real_images.size(0)
    real_targets = torch.ones(batch_size, device=device)
    fake_targets = torch.zeros(batch_size, device=device)

    discriminator.zero_grad(set_to_none=True)
    d_real_images = diff_augment(real_images, args.diffaugment_policy) if args.diffaugment else real_images
    real_logits = discriminator(d_real_images)
    if args.loss == "lsgan":
        d_loss_real = 0.5 * (real_logits - real_targets).pow(2).mean()
    else:
        d_loss_real = criterion(real_logits, real_targets)

    z = sample_noise(batch_size, args.latent_dim, device)
    fake_images = generator(z)
    d_fake_images = diff_augment(fake_images.detach(), args.diffaugment_policy) if args.diffaugment else fake_images.detach()
    fake_logits = discriminator(d_fake_images)
    if args.loss == "lsgan":
        d_loss_fake = 0.5 * (fake_logits - fake_targets).pow(2).mean()
    else:
        d_loss_fake = criterion(fake_logits, fake_targets)
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()

    generator.zero_grad(set_to_none=True)
    g_fake_images = diff_augment(fake_images, args.diffaugment_policy) if args.diffaugment else fake_images
    fake_logits_for_g = discriminator(g_fake_images)
    if args.loss == "lsgan":
        g_loss = 0.5 * (fake_logits_for_g - real_targets).pow(2).mean()
    else:
        g_loss = criterion(fake_logits_for_g, real_targets)
    g_loss.backward()
    g_optimizer.step()
    if ema is not None:
        ema.update(generator)

    return d_loss.item(), g_loss.item()


def train_wgan_gp_batch(real_images, generator, discriminator, g_optimizer, d_optimizer, args, device, step):
    batch_size = real_images.size(0)

    discriminator.zero_grad(set_to_none=True)
    z = sample_noise(batch_size, args.latent_dim, device)
    fake_images = generator(z).detach()
    real_score = discriminator(real_images).mean()
    fake_score = discriminator(fake_images).mean()
    gp = gradient_penalty(discriminator, real_images, fake_images, device)
    d_loss = fake_score - real_score + args.gp_lambda * gp
    d_loss.backward()
    d_optimizer.step()

    g_loss_value = None
    if step % args.n_critic == 0:
        generator.zero_grad(set_to_none=True)
        z = sample_noise(batch_size, args.latent_dim, device)
        generated = generator(z)
        g_loss = -discriminator(generated).mean()
        g_loss.backward()
        g_optimizer.step()
        g_loss_value = g_loss.item()

    return d_loss.item(), g_loss_value


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(vars(args), out_dir / "config.json")

    dataloader = build_dataloader(args)
    generator, discriminator = make_models(args.latent_dim, args.g_channels, args.d_channels)
    generator.to(device)
    discriminator.to(device)
    ema = EMA(generator, decay=args.ema_decay) if args.ema else None

    if args.gan_type == "wgan-gp":
        g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.9))
        criterion = None
    else:
        g_optimizer = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        criterion = nn.BCEWithLogitsLoss()

    fixed_noise = sample_noise(64, args.latent_dim, device)
    save_real_grid(dataloader, out_dir / "samples" / "real_cifar10.png")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        d_losses = []
        g_losses = []
        progress = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for batch_idx, (real_images, _) in enumerate(progress, start=1):
            real_images = real_images.to(device, non_blocking=True)
            if args.gan_type == "wgan-gp":
                d_loss, g_loss = train_wgan_gp_batch(
                    real_images, generator, discriminator, g_optimizer, d_optimizer, args, device, global_step
                )
                global_step += 1
                if g_loss is not None:
                    g_losses.append(g_loss)
            else:
                d_loss, g_loss = train_dcgan_step(
                    real_images, generator, discriminator, g_optimizer, d_optimizer, criterion, args, device, ema=ema
                )
                g_losses.append(g_loss)

            d_losses.append(d_loss)
            shown_g_loss = g_losses[-1] if g_losses else 0.0
            progress.set_postfix(d_loss=f"{d_loss:.3f}", g_loss=f"{shown_g_loss:.3f}")

            if args.fast_dev_run and batch_idx >= 4:
                break

        mean_d_loss = sum(d_losses) / max(1, len(d_losses))
        mean_g_loss = sum(g_losses) / max(1, len(g_losses))
        row = {
            "epoch": epoch,
            "d_loss": f"{mean_d_loss:.6f}",
            "g_loss": f"{mean_g_loss:.6f}",
            "seconds": f"{time.time() - start:.2f}",
            "gan_type": args.gan_type,
        }
        append_history(out_dir / "history.csv", row)
        print(
            f"Epoch {epoch:03d}: d_loss={mean_d_loss:.4f}, "
            f"g_loss={mean_g_loss:.4f}, time={row['seconds']}s"
        )

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            sample_generator = ema.ema_model if ema is not None else generator
            sample_generator.eval()
            with torch.no_grad():
                samples = sample_generator(fixed_noise).cpu()
            save_sample_grid(samples, out_dir / "samples" / f"epoch_{epoch:03d}.png")
            generator.train()

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
