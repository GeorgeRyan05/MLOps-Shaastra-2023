FROM python:3.10.7

RUN pip install --upgrade pip
RUN pip install --no-cache mlflow
RUN pip install --no-cache numpy
RUN pip install --no-cache torch torchvision --index-url https://download.pytorch.org/whl/cpu
# Saves space - Linux defaults to installing pytorch with cuda (useless for our files and useless if you don't have a Nvidia GPU)
RUN pip install --no-cache flask
ADD app.py .
ADD fashion_mnist.py .
RUN mkdir -p FashionMNIST
ADD FashionMNIST ./FashionMNIST
ADD templates ./templates

CMD ["python", "./app.py"]
