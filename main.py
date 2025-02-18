import torch
import torch.optim as optim
from data import load_mnist
from dae import DAE
from train import train

#Prevent access errors when trying to load the MNIST database
import ssl 
ssl._create_default_https_context = ssl._create_unverified_context

def main():

    #Load the dataloaders for the training and validation sets
    print("Loading training and validation sets from MNIST database...")
    train_loader, val_loader = load_mnist()

    #Initialise DAE for black and white images (hence the input depth = 1)
    myDAE = DAE(input_depth=1, filter_no=32)

    #Hyperparameters of the model. Use the ADAM optimizer.
    learning_rate = 1e-3
    n_epochs = 10
    optimizer = optim.Adam(myDAE.parameters(), lr = learning_rate)

    print("Beginning training loop...")
    #Train the model using SGD.
    train(
        n_epochs=n_epochs,
        dae=myDAE,
        optimizer=optimizer,
        train_loader=train_loader
        )

    torch.save(myDAE.state_dict(), './trained_models/first_dae.pt')


if __name__ == "__main__":
    main()