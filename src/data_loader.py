from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from .config import data, DEVICE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

def get_mnist_loaders(batch_size=None):
    if batch_size is None:
        batch_size = data['batch_size']

    pin_memory = DEVICE.type == "cuda"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transform
    )

    test_data = datasets.MNIST(
        root=str(DATA_DIR),
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=data['shuffle'],
        num_workers=data['num_workers'],
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_data,
        batch_size=1000,
        shuffle=False,
        num_workers=data['num_workers'],
        pin_memory=pin_memory
    )

    return train_loader, test_loader


