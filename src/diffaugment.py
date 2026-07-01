import torch
from torch.nn import functional as F


def rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) - 0.5)


def rand_saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) * 2.0
    return (x - mean) * scale + mean


def rand_contrast(x):
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) + 0.5
    return (x - mean) * scale + mean


def rand_translation(x, ratio=0.125):
    shift_x = int(x.size(2) * ratio + 0.5)
    shift_y = int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, (x.size(0), 1, 1), device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, (x.size(0), 1, 1), device=x.device)
    grid_x = torch.arange(x.size(2), device=x.device).view(1, -1, 1) + translation_x + shift_x
    grid_y = torch.arange(x.size(3), device=x.device).view(1, 1, -1) + translation_y + shift_y
    x_pad = F.pad(x, (shift_y, shift_y, shift_x, shift_x))
    batch = torch.arange(x.size(0), device=x.device).view(-1, 1, 1)
    return x_pad.permute(0, 2, 3, 1)[batch, grid_x, grid_y].permute(0, 3, 1, 2)


def rand_cutout(x, ratio=0.5):
    cutout_h = int(x.size(2) * ratio + 0.5)
    cutout_w = int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_h % 2), (x.size(0), 1, 1), device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_w % 2), (x.size(0), 1, 1), device=x.device)
    grid_x = torch.arange(x.size(2), device=x.device).view(1, -1, 1)
    grid_y = torch.arange(x.size(3), device=x.device).view(1, 1, -1)
    lower_x = offset_x - cutout_h // 2
    upper_x = offset_x + (cutout_h + 1) // 2
    lower_y = offset_y - cutout_w // 2
    upper_y = offset_y + (cutout_w + 1) // 2
    mask = (grid_x < lower_x) | (grid_x >= upper_x) | (grid_y < lower_y) | (grid_y >= upper_y)
    return x * mask.unsqueeze(1).to(dtype=x.dtype)


def diff_augment(x, policy="color,translation,cutout"):
    if not policy:
        return x
    for item in policy.split(","):
        item = item.strip()
        if item == "color":
            x = rand_brightness(x)
            x = rand_saturation(x)
            x = rand_contrast(x)
        elif item == "translation":
            x = rand_translation(x)
        elif item == "cutout":
            x = rand_cutout(x)
        elif item:
            raise ValueError(f"Unknown DiffAugment policy: {item}")
    return x.contiguous()
