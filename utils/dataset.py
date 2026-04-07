import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_loader(batch_size):
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, transform=transform, download=True
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,   # IMPORTANT for Windows
        pin_memory=True
    )

    return loader