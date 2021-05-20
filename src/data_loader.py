import numpy as np
import os
import torch
import zipfile
from PIL import Image
from torchvision import datasets
from torchvision import transforms

class PhotoDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        super(PhotoDataset, self).__init__()
        self.paths = list(map(lambda p: os.path.join(path, p), os.listdir(path)))
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(Image.open(self.paths[index]).convert('RGB')), 1

    def __len__(self):
        return len(self.paths)

class WashInkDataset(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        super(WashInkDataset, self).__init__()
        self.paths = list(map(lambda p: os.path.join(path, p), os.listdir(path)))
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(Image.open(self.paths[index]).convert('RGB')), 1

    def __len__(self):
        return len(self.paths)

def get_loader(config):
    """Builds and returns Dataloader for photo and washink dataset."""

    if config.extract:
        with zipfile.ZipFile(config.zip_path, 'r') as zip_ref:
            zip_ref.extractall(config.extract_path)

    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.RandomCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    photo = PhotoDataset(config.photo_path, transform)
    washink = WashInkDataset(config.washink_path, transform)

    photo_loader = torch.utils.data.DataLoader(dataset=photo,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    washink_loader = torch.utils.data.DataLoader(dataset=washink,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)
    return photo_loader, washink_loader