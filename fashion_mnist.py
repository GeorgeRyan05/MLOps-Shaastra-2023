"""Helper functions for app.py.
"""
import numpy as np, pandas as pd

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# This emsures that our results are predictable
torch.manual_seed(0)
np.random.seed(0)

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]
batch_size = 256
n_epochs = 1
lr = 2e-3
    
# train_load = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_load = DataLoader(test_set, batch_size=batch_size, shuffle=True)
# Dataloaders split the data into mini batches. Here size is 128
class CNN(nn.Module):
    """Simple convolutional neural network"""
    def __init__(self):
        nn.Module.__init__(self)
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            # 1 input image. If we had an RGB image, it would be Conv2d(3, 32, 3, padding=1)
            # 32 output images, i.e, 32 kernels and 32 output images are produced
            nn.ReLU(),
            # The activation function
            nn.MaxPool2d(2),
            # Pooling with 2 x 2 blocks
            nn.Conv2d(32, 64, 3, padding=1),
            # Now we have those 32 images and we make 64 from them
            nn.ReLU(),
            nn.MaxPool2d(2)
            # Pooling again
        )
        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear((28*28*64)//16, 600),
            # The image shape was initially 28 x 28, by pooling we've made it 7 x 7, so we divide by 16
            # We multiply by 64 because the model has learnt 64 features.
            nn.Linear(600, 128),
            nn.Linear(128, 10)
            # We have 10 output neurons (1 for each class)
        )
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.convolutions(inputs)
        # Functions in convolution layers are run
        x = self.fully_connected(x)
        # Functions in fully connected layer are run
        return x
    @torch.inference_mode
    def predict(self, images):
        "Predict class of images"
        x: torch.Tensor = self(images)
        return x.max(1)[1]

class ANN(nn.Module):
    """Simple neural network. 
    This will perform poorly on this dataset because the dataset contains images.
    """
    def __init__(self):
        nn.Module.__init__(self)
        self.stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 600),
            nn.ReLU(),
            nn.Linear(600, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU()
        )
        # This is an ordinary neural network for comparison

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.stack(inputs)
        return x

def train(model: nn.Module, train_load: DataLoader | Dataset, optimizer=torch.optim.Adam, loss=nn.CrossEntropyLoss(), ps=None, batch_size=batch_size) -> None:
    """Train the model on the train loader and return the accuracy.
    
    Args:
        model (nn.Module): model to train.
        train_load (Dataset | DataLoader): train Dataset or DataLoader.
        optimizer (torch.optim.Optimizer): optimizer to use for updating the model's weights.
        loss (torch.nn.modules.loss._WeightedLoss): loss function to use between labels and predictions.
        ps (int | None): how often to print the accuracy. If None, no printing.
        batch_size (int): size of each batch.
    """
    if isinstance(train_load, Dataset):
        train_load = DataLoader(train_load, batch_size=batch_size, shuffle=True)
    train_iter = iter(train_load)

    optimizer = optimizer(model.parameters(), lr=lr)
    model.train()
    # Puts model in train mode
    targets: torch.Tensor
    for i, (data, targets) in enumerate(train_iter):
        # i is iteration, data = 1 mini batch of images, targets = 1 mini batch target values
        # This repeats for all mini batches 
        outputs: torch.Tensor = model.forward(data) # Forward pass
        loss_val: torch.Tensor = loss(outputs, targets) # Loss computation
        optimizer.zero_grad()  # Ensures gradients stored in optimizer are reset before each backward pass
        loss_val.backward() # Backward pass
        optimizer.step() # Backward pass

        if ps and i % ps == 0:
            model.eval()
            # Puts model in evaluation mode, so we 
            with torch.no_grad():
                print(f"Loss is {loss_val}")
                predicted = outputs.max(1)[1]
                correct = (predicted == targets).sum().item()
                accuracy = correct/len(targets)
                print(f"Train accuracy is {accuracy*100:.3f}%")

def accuracy(model: nn.Module, test: Dataset | DataLoader, batch_size=batch_size):
    """Evaluate a model given a test loader.
    
    Args:
        model (nn.Module): model to evaluate.
        test (Dataset | DataLoader): test Dataset or DataLoader.
        batch_size (int): size of each batch.
    Returns:
        Fraction of correct responses.
    """
    model.eval()
    with torch.no_grad():
        count = 0
        correct = 0
        if isinstance(test, Dataset):
            test = DataLoader(test, batch_size=batch_size, shuffle=True)
        targets: torch.Tensor
        for data, targets in iter(test):
            outputs: torch.Tensor = model(data)
            predicted = outputs.max(1)[1] # Maximum output is predicted class
            count += len(targets) # Total length of datasetS
            correct += (predicted == targets).sum().item()
            # This gives a tensor of True and False values and adds no. of True values to correct each iteration
        accuracy = correct/count
        return accuracy