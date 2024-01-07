FROM python:3.10.7

RUN pip install --upgrade pip
RUN pip install --no-cache mlflow
RUN pip install --no-cache numpy
RUN pip install --no-cache torch torchvision
RUN pip install --no-cache flask

CMD ["python", "./app.py"]