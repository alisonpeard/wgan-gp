#%%
import os
import numpy as np
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from dotenv import load_dotenv, find_dotenv

# %%
def get_era5_dataloaders(batch_size=128, img_size=(32, 32),
                         train_size=60000, test_size=10000,
                         gumbel=False, **kwargs):
    load_dotenv(find_dotenv(usecwd=True))
    path = os.getenv('PRETRAIN_PATH')
    data = np.load(path)['data']
    data = np.flip(data, axis=1) # Northern hemisphere

    all_transforms = transforms.Compose([
        transforms.Resize(img_size)
    ])

    # Generate random indices for train and test sets
    indices = np.random.permutation(len(data))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:train_size + test_size]
    train = data[train_indices, ...].astype(np.float32)
    test = data[test_indices, ...].astype(np.float32)
    train = train.reshape(len(train), 1, train.shape[1], train.shape[2])
    test = test.reshape(len(test), 1, test.shape[1], test.shape[2])
    if gumbel:
        train = -np.log(-np.log(train))
        test = -np.log(-np.log(test))
    train = tensor(train)
    test = tensor(test)

    # all_transforms(data)
    train = TensorDataset(all_transforms(train))
    test = TensorDataset(all_transforms(test))
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_mnist_dataloaders(batch_size=128, **kwargs):
    """MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.MNIST('../data', train=True, download=True,
                                transform=all_transforms)
    test_data = datasets.MNIST('../data', train=False,
                               transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_fashion_mnist_dataloaders(batch_size=128, **kwargs):
    """Fashion MNIST dataloader with (32, 32) sized images."""
    # Resize images so they are a power of 2
    all_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.FashionMNIST('../fashion_data', train=True, download=True,
                                       transform=all_transforms)
    test_data = datasets.FashionMNIST('../fashion_data', train=False,
                                      transform=all_transforms)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader


def get_lsun_dataloader(path_to_data='../lsun', dataset='bedroom_train',
                        batch_size=64, **kwargs):
    """LSUN dataloader with (128, 128) sized images.

    path_to_data : str
        One of 'bedroom_val' or 'bedroom_train'
    """
    # Compose transforms
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor()
    ])

    # Get dataset
    lsun_dset = datasets.LSUN(db_path=path_to_data, classes=[dataset],
                              transform=transform)

    # Create dataloader
    return DataLoader(lsun_dset, batch_size=batch_size, shuffle=True)
