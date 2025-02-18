######################################################################
# Module for handling the loading of the MNIST training and datasets #
######################################################################

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from typing import Tuple

def load_mnist(data_path:str='.', batch_size:int=64)->Tuple[DataLoader, DataLoader]:
    """Function that loads the MNIST database and returns the training and validation sets in dataloader form.
    
    Args
    ----
        data_path (str): The directory (as a path) where the user wants to store the MNIST data.
        batch_size (int): The desired batch size for the train_loader

    Returns
    -------
        train_loader (DataLoader): The MNIST training set
        val_loader (DataLoader): The MNIST validation set

    """

    #Load training and validation sets as DataSets. Set download to true (obvious), and make sure to transform to tensors

    train_set = datasets.MNIST(
        root=data_path, 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
        )
    
    val_set = datasets.MNIST(
        root=data_path, 
        train=True, 
        download=True, 
        transform=transforms.ToTensor()
        )
    

    #Convert to DataLoader type. Make sure to set the training set to "shuffle=True" so that we get a better cross-section
    #     of the training set during training.
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader