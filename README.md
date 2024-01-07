# MLOps-Shaastra-2023
Download the Dockerfile into a directory of your choice go to that directory (using cd or ls).
This directory will be used for the remaining part of this workshop.
Run the following command
```
docker build -t mlops-workshop .
```
Whether or not you are not able to install docker, run this also (in a virtual environment if you want). It won't run on python 3.12 
```
pip install pip
pip install mlflow
pip install numpy
pip install torch torchvision
pip install flask
```
Build:
```
docker build -t mlops-workshop .
docker run -i -p 1000:1000 -p 5000:5000 --mount type=bind,source="%cd%/mlflow_uri",target=/mlflow_uri -t mlops-workshop```
