import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.embed = nn.Embedding(num_classes, num_features * 2)
        self.embed.weight.data[:, :num_features].fill_(1.0)
        self.embed.weight.data[:, num_features:].zero_()

    def forward(self, x, labels):
        gamma, beta = self.embed(labels).chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return self.bn(x) * gamma + beta


class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, labels):
        residual = F.interpolate(x, scale_factor=2, mode="nearest")
        residual = self.skip(residual)

        out = F.relu(self.bn1(x, labels), inplace=True)
        out = F.interpolate(out, scale_factor=2, mode="nearest")
        out = self.conv1(out)
        out = F.relu(self.bn2(out, labels), inplace=True)
        out = self.conv2(out)
        return out + residual


class ResNetConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=128, num_classes=10, base_channels=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(latent_dim, base_channels * 4 * 4)
        self.block1 = GeneratorBlock(base_channels, base_channels, num_classes)
        self.block2 = GeneratorBlock(base_channels, base_channels, num_classes)
        self.block3 = GeneratorBlock(base_channels, base_channels, num_classes)
        self.bn = nn.BatchNorm2d(base_channels)
        self.conv = nn.Conv2d(base_channels, 3, 3, padding=1)

    def forward(self, z, labels):
        out = self.fc(z).view(z.size(0), -1, 4, 4)
        out = self.block1(out, labels)
        out = self.block2(out, labels)
        out = self.block3(out, labels)
        out = F.relu(self.bn(out), inplace=True)
        return torch.tanh(self.conv(out))


class OptimizedDiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.skip = spectral_norm(nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        out = F.avg_pool2d(out, 2)
        residual = F.avg_pool2d(self.skip(x), 2)
        return out + residual


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        self.downsample = downsample
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, 3, padding=1))
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, 3, padding=1))
        self.skip = spectral_norm(nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        out = F.relu(x, inplace=True)
        out = self.conv1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        residual = self.skip(x)
        if self.downsample:
            out = F.avg_pool2d(out, 2)
            residual = F.avg_pool2d(residual, 2)
        return out + residual


class ResNetProjectionDiscriminator(nn.Module):
    def __init__(self, num_classes=10, base_channels=128):
        super().__init__()
        self.block1 = OptimizedDiscriminatorBlock(3, base_channels)
        self.block2 = DiscriminatorBlock(base_channels, base_channels, downsample=True)
        self.block3 = DiscriminatorBlock(base_channels, base_channels, downsample=True)
        self.block4 = DiscriminatorBlock(base_channels, base_channels, downsample=False)
        self.linear = spectral_norm(nn.Linear(base_channels, 1))
        self.embed = spectral_norm(nn.Embedding(num_classes, base_channels))

    def forward(self, images, labels):
        out = self.block1(images)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = F.relu(out, inplace=True)
        feat = out.sum(dim=(2, 3))
        unconditional = self.linear(feat).squeeze(1)
        projection = torch.sum(self.embed(labels) * feat, dim=1)
        return unconditional + projection


def init_resnet_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear, nn.Embedding)):
        nn.init.xavier_uniform_(module.weight)
        if getattr(module, "bias", None) is not None:
            nn.init.zeros_(module.bias)
