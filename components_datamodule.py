import os
import torch
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as T
from PIL import Image
from typing import Optional
from torch.utils.data import random_split, Dataset, DataLoader


class ComponentsDataModule(pl.LightningDataModule):
    def __init__(self, root_path, batch_size, num_workers=4, transforms=None):
        super().__init__()
        self.root_path = root_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        if self.transforms is None:
            transforms = T.Compose([
                T.Resize((112, 112)),
                T.ColorJitter(brightness=.5, hue=.3),
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                T.RandomSolarize(threshold=192.0),
                T.ToTensor(),
                T.Normalize(mean=(0.2979, 0.2789, 0.2408), std=(0.2960, 0.2848, 0.2620))
            ])
            self.transforms = transforms

    def prepare_data(self) -> None:
        # Tasks like downloading data
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        components_ds = ComponentsDataset(self.root_path, self.transforms)
        size_components_ds = len(components_ds)
        train_size = int(size_components_ds * 0.7)
        val_size = int(size_components_ds * 0.15)
        test_size = size_components_ds - (train_size + val_size)
        self.train_ds, self.val_ds, self.test_ds = random_split(
            components_ds, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )


class ComponentsDataset(Dataset):
    """
    Same function as ImageFolder in pytorch
    """
    def __init__(self, root_path, transforms=None) -> None:
        super().__init__()
        self.root_path = root_path
        self.transforms = transforms
        classes, class_idx_dict = self.class_to_idx(root_path)
        print(class_idx_dict)
        self.classes = classes
        self.class_idx_dict = class_idx_dict
        self.samples = self.get_samples(root_path, class_idx_dict)
        self.targets = [sample[1] for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, target_id = self.samples[index]
        image = Image.open(img_path)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target_id

    def get_samples(self, root_path, class_to_idx):
        directory = os.path.expanduser(root_path)

        samples = []
        available_classes = set()

        for target_class in sorted(class_to_idx.keys()):
            target_id = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, file_names in sorted(os.walk(target_dir, followlinks=True)):
                for file_name in file_names:
                    file_dir = os.path.join(root, file_name)
                    sample = file_dir, target_id
                    samples.append(sample)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"No files found for the following classes {' , '.join(empty_classes)}"
            raise FileNotFoundError(msg)

        return samples

    def class_to_idx(self, root_path):
        classes = [directory.name for directory in os.scandir(
            root_path) if directory.is_dir()]
        class_idx_dict = {name: i for i, name in enumerate(classes)}
        return classes, class_idx_dict


if __name__ == "__main__":
    root_path = '/storage/PCB-Components-L1'
    components_ds = ComponentsDataset(root_path)
    it = iter(components_ds)
    next(it)
    print(f"Dataset len { len(components_ds) }\n")

    components_dm = ComponentsDataModule(root_path, 32, 8)
