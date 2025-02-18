from torch.utils.data import DataLoader
import torch.optim as optim
from dae import DAE
import time


def train(
        n_epochs:int,
        dae:DAE,
        optimizer:optim.Optimizer,
        train_loader:DataLoader
    )->None:
    """
    Function responsible for training the model. 

    Parameters
    ----------
        n_epochs : int
            The number of training loops we want to run gradient descent for.
        dae : DAE
            The denoising autoencoder to be trained via some optimization algorithm.
        optimizer : torch.optim.Optimizer
            The optimizer for the model.
        train_loader : DataLoader
            The dataloader containing the training set.
    
    """

    for epoch in range(1, n_epochs+1):
        t_i = time.time()

        #Initialise the loss to 0 at the beginning of each epoch
        loss = 0.0

        for imgs, labels in train_loader:

            #FORWARDS PASS
            outputs = dae(imgs) #output image tensors
            loss_train = dae.loss_fn(imgs, outputs) #calculate loss function
            loss += loss_train #add loss using the DAE's preferred loss function

            #BACKWARDS PASS
            optimizer.zero_grad() #clear gradients to prevent accumulation
            loss_train.backward() #perform backwards pass on the network
            optimizer.step() #perform gradient descent

        mean_batch_loss = loss / len(train_loader)

        t_f = time.time()

        print(f"Mean batch loss at epoch {epoch}: {mean_batch_loss}. Epoch length: {t_f-t_i} seconds.")

    
    return 0