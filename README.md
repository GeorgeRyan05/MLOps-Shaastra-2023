# Installation
1. Install Docker from the https://www.docker.com/get-started/
    - If this doesn't work properly, install WSL 2.
    - If that doesn't work, check https://learn.microsoft.com/en-us/windows/wsl/troubleshooting#installation-issues
2. As of now, this does not work on python 3.12 (due to pytorch not being updated yet). The docker version should work. Otherwise, you can run in a virual environment.
# Running with Docker
Navigate to the directory with docker.  
You can use any name instead of mlops-workshop. If you want to name your mlflow storage something else, you will need to change the code (`mlflow.set_tracking_uri()`). 
```
docker build -t mlops-workshop .
docker run -i -p 1000:1000 -p 5000:5000 --mount type=bind,source="%cd%/mlflow_uri",target=/mlflow_uri -t mlops-workshop
```
If the 1<sup>st</sup> command doesn't work, run  

```
docker build -f "%cd%/Dockerfile" -t mlops-workshop .
docker run -i -p 1000:1000 -p 5000:5000 --mount type=bind,source="%cd%/mlflow_uri",target=/mlflow_uri -t mlops-workshop
```
The app will be served on your browser at http://127.0.0.1:1000
# Running outside of Docker
The following command should work to get packages. (this can be run in a virual environment if you prefer)
```
pip install mlflow
pip install numpy
pip install torch
pip install torchvision
pip install flask
```
Run the app:
```
python app.py
```
# Some Errors
You might get some errors that are a bit hard to diagnose when trying docker and mlflow out for yourself, I thought I'd mention some of them. I don't think these will occur with the above commands.

You can't reuse the same ***experiment name*** when running locally and on docker. This is for of a few reasons:
- Docker runs in a virtual environment, so the MLflow uri absolute path is just /mlflow_uri, whereas on your system it would be C:/Users/.../mlflow_uri
- MLflow experiments store the directory of the experiment in <experiment_id>/meta.yaml.
- If you start an experiment locally, it won't work on docker (throws an error)
- So, if you started the experiment on docker and then try to run locally, when you save a new model, it will be saved in experiment_directory/run_directory, even though the experiment directory is /mlflow_uri. This saves it in C:/mlflow_uri, which docker can no longer access (if you try to run it in docker again).  
**BUT**, it still registers the model under the correct uri, with the actual (unexpected) location, so an error won't appear when running locally (but will if you switch back to docker). Under ./mlflow_uri/models, the model's full path is saved (in meta.yaml). (./mlflow_uri/models is always in the right place, though the full path may point outside that directory.)
In fact, both docker and local files will work, as long as you save a model with them. 
 
TLDR: It will seem to work partially, but it's probably not what you are expecting and you'll have to repeat training. Use a different experiment name (or better still, a different backend URI) when running on docker.

For the same reason, mlflow ui --backend-store-uri ./mlflow_uri will show several file not found errors when you try to view it locally if you've previously run it on docker and vice versa.

Not using --host 0.0.0.0 (for both flask and mlflow) can cause issues when running on docker. Also, even when using host 0.0.0.0, mlflow will show the wrong link in the command line. (The actual link will be http://127.0.0.1:5000)

## Viewing MLflow in Docker
The solution for viewing mlflow in docker is a little odd.  
It works because flask essentially lets you run multiple functions at the same time, by allowing one to run on a different URL. (This doesn't actually require you to keep the URL open in a tab, flask will keep running anyway.)  
This kind of solution is needed because the mlflow command won't finish running until the app is closed.
I believe you could also implement something using asynchronous programming.

## Reclaiming space
Docker works with Linux and partitions your storage, so deleting containers doesn't actually return your storage.  
To get your storage back, you can try restrating Docker Desktop and see if that works, otherwise go to Docker Desktop and clear all data.  