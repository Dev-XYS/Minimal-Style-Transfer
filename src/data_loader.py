import numpy as np
import os
import torch
from mlxtend.data import loadlocal_mnist
from PIL import Image
from torchvision import datasets
from torchvision import transforms

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, input, label):
        super(MNISTDataset, self).__init__()
        self.input = input
        self.label = label

    def __getitem__(self, index):
        return transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])(Image.fromarray(self.input[index])), self.label[index]
        
    def __len__(self):
        return len(self.input)

def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""
    
    transform = transforms.Compose([
                    transforms.Scale(config.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    svhn = datasets.SVHN(root=config.svhn_path, download=False, transform=transform)
    X, y = loadlocal_mnist(
        images_path=os.path.join(config.mnist_path, 'MNIST/raw/train-images-idx3-ubyte'),
        labels_path=os.path.join(config.mnist_path, 'MNIST/raw/train-labels-idx1-ubyte')
    )
    X = np.reshape(X, (len(X), 28, 28))
    mnist = MNISTDataset(X, y)

    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)
    return svhn_loader, mnist_loader