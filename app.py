"""This file serves the web app.
"""
import mlflow
from PIL import Image, ImageOps
from fashion_mnist import class_names, CNN, ANN, train as train_, accuracy
from torchvision import transforms, datasets
import os
from flask import Flask, request, render_template

# All print statements in this file appear only in the terminal, for debugging purposes. They will not show up on the web app

app = Flask(__name__) # Required for Flask
mlflow.set_tracking_uri("./mlflow_uri") # Storage directory for mlflow.
# This should be passed as a bind mount to docker run.
mlflow.set_experiment('FashionMNIST for MLOps')
mlflow.pytorch.autolog(disable=True)

transform = transforms.Compose([
                transforms.Resize((28,28)),
                transforms.ToTensor(),
                transforms.Grayscale(),
                transforms.Normalize((0,), (1,))
])
# Does normalization of data for us, removes colour and ensures the data is in the correct shape and data type.
def train():
    """Train the model on the FashionMNIST dataset."""
    # Loading the Fashion MNIST dataset
    train_set = datasets.FashionMNIST('.', train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST('.', train=False, download=True, transform=transform) 
    # The data is downloaded only the 1st time

    # Running the model
    with mlflow.start_run(run_name='FashionMNIST'):
        mlflow.set_tag('model_name', 'CNN')
        mlflow.set_tag('training', True)

        model = CNN()
        train_(model, batch_size=256, train_load=train_set) 
        # Train the model using our module fashion_mnist.py
        acc = accuracy(model, test_set) # Test the model

        mlflow.log_params({'epochs': 1, 'batch_size': 256}) # Log the hyperparameters
        mlflow.log_metric('accuracy', acc) # Log the accuracy
        mlflow.pytorch.log_model(model, 'FashionMNIST CNN', registered_model_name='FashionMNIST CNN') # Saves the model

# The below function is run when this script is run. (on app.run())
@app.route('/', methods=['GET', 'POST'])
def home():
    "Displays the UI"
    if request.method == 'POST':
        # This runs when a form is submitted in home.html
        print('Getting response')
        img = request.files['image'].stream # File input
        mode = request.form.get('train') or request.form.get('classify')
        # Depending on which button is pressed.
        if mode == 'classify':
            print('Got mode')
            answer = classify(img)
            print('classified')
            return render_template('home.html', answer=answer)
            # home.html is jinja2 template, so it is a html file, which also lets us take some input.
        if mode == 'train':
            train() # Trains our model.
            return render_template('home.html', training='done')
    return render_template('home.html') # This occurs if we have a GET request

# Another page in our app. This one will freeze the rest of the app temporarily.
@app.route('/mlflow', methods=['GET'])
def mlflow_ui():
    """When this function runs, the app will freeze. 
    The mlflow UI will start, but not open. 
    There is a link beside this to open it in home.html (or you can visit http://127.0.0.1:5000)

    To continue using the app, reload the 1st page (http://127.0.0.1:1000 )
    """
    os.system('mlflow ui --host 0.0.0.0 --backend-store-uri ./mlflow_uri')
    # This will not stop running until the app is closed

def classify(image_path):
    "Classify an image given its path or a file object (here we use a file object)."
    print('Running classify')
    img_raw = Image.open(image_path)
    img = transform(ImageOps.grayscale(img_raw)).unsqueeze(0) 
    # PyTorch models expect a batch of inputs
    with mlflow.start_run(run_name='FashionMNIST'):
        mlflow.set_tag('model_name', 'CNN')
        mlflow.set_tag('training', False)
        mlflow.log_image(img_raw, 'Image.png') 
        # Save the image to the mlflow storage, so we can see inputs for a run.
        # Note that the model will see a much lower resolution version of the image and in black and white.

        pre_trained_model = mlflow.pytorch.load_model('models:/FashionMNIST CNN/latest')
        output = pre_trained_model.predict(img)

        mlflow.set_tag('output', class_names[output]) # Log the output
    print(class_names[output])
    return class_names[output]

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=1000)
    # Runs home()