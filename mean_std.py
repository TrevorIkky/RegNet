import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader
from components_datamodule import ComponentsDataset

def get_mean_and_std(data_loader):
    # var(x) = E(x**2) - E(x)**2
    channels_sum, channels_sum_sq, n_batches = 0, 0, 0

    for img, _ in data_loader:
        channels_sum += torch.mean(img, dim=[0, 2, 3])
        channels_sum_sq += torch.mean(img ** 2, dim=[0, 2, 3])
        n_batches += 1

    mean = channels_sum / n_batches
    std = (channels_sum_sq / n_batches - mean ** 2) ** 0.5

    return mean, std

if __name__ == "__main__":
    root_path = '/storage/PCB-Components-L1'
    transforms = T.Compose([
        T.Resize((112, 112)),
        T.ToTensor()
    ])
    
    dataset = ComponentsDataset(root_path, transforms)
    data_loader = DataLoader(dataset, batch_size=128, num_workers=8, shuffle=True)
    mean, std = get_mean_and_std(data_loader)
    print(f"Mean is: {mean} \nStd is: {std}")