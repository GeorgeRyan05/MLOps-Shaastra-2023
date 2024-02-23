"""Example main.py file for running in an interactive terminal.

This uses docker but not flask.
You can change the CMD in the Dockerfile to run this instead of app.py.
There is no need to connect a port if using this.
This will not work with the mlflow display and is only intended to teach you how to use Docker.
"""
from . import main
import mlflow
from PIL import Image, ImageOps
import numpy as np
from fashion_mnist import class_names
from torchvision import transforms
import os
print(os.listdir('.'))
print(os.listdir('mlflow_uri'))

mlflow.set_tracking_uri("./mlflow_uri")
mode = input('What do you want to do? train or run\n')

transform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Normalize((0,), (1,))
])
# Does normalization of data for us and ensures the data is in the correct shape and data type

while mode in {'train', 'run'}:
    if mode == 'train':
        from fashion_mnist import CNN, ANN, train, accuracy
        from torchvision import datasets

        mlflow.set_experiment('FashionMNIST for MLOps')
        mlflow.pytorch.autolog(disable=True)
        # Loading the Fashion MNIST dataset

        train_set = datasets.FashionMNIST('.', train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST('.', train=False, download=True, transform=transform) 
        # The data is downloaded only the 1st time
        with mlflow.start_run(run_name='FashionMNIST'):
            mlflow.set_tag('model_name', 'CNN')
            mlflow.set_tag('training', True)
            model = CNN()
            train(model, batch_size=256, train_load=train_set)
            acc = accuracy(model, test_set)

            mlflow.log_params({'epochs': 1, 'batch_size': 256})
            mlflow.log_metric('accuracy', acc)
            mlflow.pytorch.log_model(model, 'FashionMNIST CNN', registered_model_name='FashionMNIST CNN')

    if mode == 'run':
        image_path = input('Enter image path: ')
        img_raw = Image.open(image_path)
        img = transform(ImageOps.grayscale(img_raw)).unsqueeze(0)
        mlflow.set_experiment('FashionMNIST for MLOps')
        mlflow.pytorch.autolog(disable=True) 

        with mlflow.start_run(run_name='FashionMNIST'):
            mlflow.set_tag('model_name', 'CNN')
            mlflow.set_tag('training', False)
            mlflow.log_image(img_raw, 'Image.png')
            pre_trained_model = mlflow.pytorch.load_model('models:/FashionMNIST CNN/3')
            output = pre_trained_model.predict(img)
            mlflow.set_tag('output', class_names[output])
        print(class_names[output])
    mode = input('What do you want to do? train or run ?\n')