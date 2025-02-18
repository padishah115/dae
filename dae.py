import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

class DAE(nn.Module):
    """Denoising autoencoder class, following the architecture described in Venkataraman (2022).
    
    Attributes
    ----------
        forward : torch.Tensor 
    
    """

    def __init__(
            self, 
            input_depth:int=1,
            filter_no = 32
            ):
        
        """
        Parameters
        ----------
            input_depth : int 
                The number of channels in the input image tensor- this should be 3 for RGB images, 1 for B&W.
            filter_no : int 
                The number of filters (and therefore number of features after convolution) that we want in the network.

        """

        super().__init__() #call initialisation function on parent class

        #Set the loss function. We're going to use MSE loss because that's what Venkataraman used.
        self.loss_fn = nn.MSELoss()

        #Convolution and pooling layer
        self.conv1 = nn.Conv2d(in_channels=input_depth, out_channels=filter_no, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=filter_no, out_channels=filter_no, kernel_size=3, padding=1)

        #Transpose convolution layer, which both times doubles the dimensions of the feature map
        self.transconv = nn.ConvTranspose2d(in_channels=filter_no, out_channels=filter_no, kernel_size=2, stride=2)

        #Final convolutional layer, which reduces the channel number back to what it was.
        self.final_conv = nn.Conv2d(in_channels=filter_no, out_channels=input_depth, kernel_size=3, padding=1)
        

    def forward(self, x):
        """Network forward pass for the Convolutional Denoising Autoencoder.
        
        Parameters
        ----------
            x : torch.Tensor
                Flattened input image tensor to be passed through the network.

        Returns
        -------
            x : torch.Tensor
                Flattened output image tensor to be returned by the network.
        """

        #Apply convolution/maxpool block twice
        x = F.max_pool2d(input=torch.relu(self.conv1(x)), kernel_size=2)
        x = F.max_pool2d(input=torch.relu(self.conv2(x)), kernel_size=2)

        #Apply transverse convolution twice
        x = torch.relu(self.transconv(x))
        x = torch.relu(self.transconv(x))

        #Apply final convolutional layer
        x = torch.relu(self.final_conv(x))

        return x

