from pathlib import Path
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


CIFAR10_CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class LocalCIFAR10(Dataset):
    """Load CIFAR-10 directly from an extracted cifar-10-batches-py folder."""

    def __init__(self, data_dir, train=True, transform=None, max_items=None):
        self.data_dir = Path(data_dir)
        self.train = train
        self.transform = transform

        if not self.data_dir.exists():
            raise FileNotFoundError(f"CIFAR-10 directory not found: {self.data_dir}")

        batch_names = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]
        images = []
        labels = []
        for name in batch_names:
            path = self.data_dir / name
            if not path.exists():
                raise FileNotFoundError(f"Missing CIFAR-10 batch file: {path}")
            with path.open("rb") as handle:
                batch = pickle.load(handle, encoding="latin1")
            images.append(batch["data"])
            labels.extend(batch.get("labels", batch.get("fine_labels")))

        array = np.concatenate(images, axis=0)
        array = array.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.images = array
        self.labels = np.asarray(labels, dtype=np.int64)

        if max_items is not None:
            self.images = self.images[:max_items]
            self.labels = self.labels[:max_items]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index])
        label = int(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label
