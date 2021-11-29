
import pytorch_lightning as pl

from typing import Optional
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split, DataLoader

class Cifar10DataModule(pl.LightningDataModule):
    def __init__(self, data_dir:str='./dataset', batch_size:int=32, num_workers:int=8, download=True):
        super().__init__()

        #dataset specific items
        self.num_classes = 10
        self.image_dims = (3, 112, 112)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.download = download

        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))
        ])

    def prepare_data(self):
        #Do tasks such as download data
        CIFAR10(self.data_dir, train=True, download=self.download)
        CIFAR10(self.data_dir, train=False, download=self.download)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.cifar_train, self.cifar_val = random_split(
                CIFAR10(self.data_dir, train=True, transform=self.transform),
                [45000, 5000]
            )
        if stage == 'test' or stage is None:
            self.cifar_test = CIFAR10(self.data_dir, train=False, transform=self.transform)


    def train_dataloader(self):
        return DataLoader(
            self.cifar_train, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.cifar_val, batch_size=self.batch_size,
            num_workers=self.num_workers, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.cifar_test, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers
        )

