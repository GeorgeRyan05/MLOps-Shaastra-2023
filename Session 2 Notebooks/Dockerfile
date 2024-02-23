FROM python:3.10.7

RUN pip install --upgrade pip
RUN pip install --no-cache mlflow
RUN pip install --no-cache numpy
RUN pip install --no-cache torch torchvision
RUN pip install --no-cache flask
# RUN apt-get update && apt-get install libgl1 -y
ADD app.py .
ADD fashion_mnist.py .
RUN mkdir -p FashionMNIST
ADD FashionMNIST ./FashionMNIST
# ADD Sumedh.png .
ADD templates ./templates

CMD ["python", "./app.py"]
