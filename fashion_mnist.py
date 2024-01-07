# %%
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn

# %%
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])


# %%
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# %%

# %%
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear((28*28*64)//16, 600),
            nn.Linear(600, 128),
            nn.Linear(128, 10)
        )
    def forward(self, inputs):
        x = self.convolutions(inputs)
        x = self.fully_connected(x)
        return x
    @torch.inference_mode
    def predict(self, images):
        x = self(images)
        return x.max(1)[1]


# %%
def train(model, train_load, optimizer=torch.optim.Adam, loss=nn.CrossEntropyLoss()):
    train_iter = iter(train_load)
    optimizer = optimizer(model.parameters(), lr=1e-3)
    for data, targets, in train_iter:
        model.train()
        outputs = model(data)
        optimizer.zero_grad()
        loss_val = loss(outputs, targets)
        loss_val.backward()
        optimizer.step()
def accuracy(model, test_load):
    model.eval()
    test_iter = iter(test_load)
    correct = 0
    count = 0
    for data, targets in iter(test_iter):
        outputs = model(data)
        predicted = outputs.max(1)[1]
        correct = correct + (predicted == targets).sum().item()
        count = count + len(targets)
    return correct/count

# %%
# %%



