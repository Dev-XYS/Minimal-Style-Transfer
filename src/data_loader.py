import numpy as np
import os
import torch
from PIL import Image
from torchvision import transforms

class PhotoDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        super(PhotoDataset, self).__init__()
        self.transform = transform
        self.paths = paths

    def __getitem__(self, index):
        return self.transform(Image.open(self.paths[index]).convert('RGB')), 1

    def __len__(self):
        return len(self.paths)

class WashInkDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        super(WashInkDataset, self).__init__()
        self.transform = transform
        self.paths = paths

    def __getitem__(self, index):
        return self.transform(Image.open(self.paths[index]).convert('RGB')), 1

    def __len__(self):
        return len(self.paths)

def get_images_paths(path):
    return list(map(lambda p: os.path.join(path, p), filter(lambda p: not p.endswith(".webp"), os.listdir(path))))

def get_loader(config):
    """Builds and returns Dataloader for photo and washink dataset."""

    photo_imgs = get_images_paths(config.photo_path)
    washink_imgs = get_images_paths(config.washink_path)

    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.RandomCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    photo = PhotoDataset(photo_imgs, transform)
    washink = WashInkDataset(washink_imgs, transform)
    print('Initialized datasets')

    photo_loader = torch.utils.data.DataLoader(dataset=photo,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    washink_loader = torch.utils.data.DataLoader(dataset=washink,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)
    print('Initialized dataloaders')
    return photo_loader, washink_loader