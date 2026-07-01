import torch
from torch import nn
from torch.nn.utils import spectral_norm


def weights_init(module):
    classname = module.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, latent_dim=128, base_channels=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        if z.ndim == 2:
            z = z[:, :, None, None]
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base_channels, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, 4, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.features(x).flatten(1).squeeze(1)


def make_models(latent_dim=128, g_channels=64, d_channels=64):
    generator = Generator(latent_dim=latent_dim, base_channels=g_channels)
    discriminator = Discriminator(base_channels=d_channels)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    return generator, discriminator


def sample_noise(batch_size, latent_dim, device):
    return torch.randn(batch_size, latent_dim, device=device)


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10, label_dim=32, base_channels=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embed = nn.Embedding(num_classes, label_dim)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + label_dim, base_channels * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        label_vec = self.embed(labels)
        x = torch.cat([z, label_vec], dim=1)
        return self.net(x[:, :, None, None])


class ProjectionDiscriminator(nn.Module):
    def __init__(self, num_classes=10, base_channels=64):
        super().__init__()
        self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(3, base_channels, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.head = spectral_norm(nn.Linear(base_channels * 4 * 4 * 4, 1))
        self.embed = spectral_norm(nn.Embedding(num_classes, base_channels * 4 * 4 * 4))

    def forward(self, images, labels):
        feat = self.features(images).flatten(1)
        unconditional = self.head(feat).squeeze(1)
        projection = torch.sum(self.embed(labels) * feat, dim=1)
        return unconditional + projection
