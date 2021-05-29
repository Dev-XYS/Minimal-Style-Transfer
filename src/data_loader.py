import numpy as np
import os
import compress_pickle as pickle
import torch
import zipfile
from PIL import Image
from torch.functional import split
from torchvision import datasets
from torchvision import transforms

class PhotoDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform):
        super(PhotoDataset, self).__init__()
        self.transform = transform
        self.images = list(map(transform, images))

    def __getitem__(self, index):
        return self.images[index], 1

    def __len__(self):
        return len(self.images)

class WashInkDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform):
        super(WashInkDataset, self).__init__()
        self.transform = transform
        self.images = list(map(transform, images))

    def __getitem__(self, index):
        return self.images[index], 1

    def __len__(self):
        return len(self.images)

def get_images(path):
    images = []
    for p in os.listdir(path):
        if p.endswith(".webp"):
            continue
        p = os.path.join(path, p)
        images.append(Image.open(p).convert('RGB'))
    return images

def get_loader(config):
    """Builds and returns Dataloader for photo and washink dataset."""

    # photo_imgs = pickle.load(config.photo_path)
    # print('Loaded pickle file 1')
    # washink_imgs = pickle.load(config.washink_path)
    # print('Loaded pickle files')

    photo_imgs = get_images(config.photo_path)
    washink_imgs = get_images(config.washink_path)

    transform = transforms.Compose([
        transforms.Resize(config.image_size),
        transforms.RandomCrop(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # transform = lambda x: x

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