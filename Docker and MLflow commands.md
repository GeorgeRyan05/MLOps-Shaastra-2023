# Docker

```
docker build -t <image-name> <directory>

docker run -i -p 1000:1000 -p 5000:5000 --mount type=bind,source="%cd%/mlflow_uri",target=/mlflow_uri -t <image-name>
```
#### .dockerignore
If you want to ignore certain files completely, you can use a .dockerignore file (It's like .gitignore). This shouldn't be needed if individually adding files/folders with ADD.
### docker build:
* -t : Name the image
* -f : Enter location of Dockerfile (usually defaults to ./Dockerfile)

Finally enter the directory (relative or absolute) containing the Dockerfile (or use -f to specify the dockerfile location). All the RUN and ADD commands will be done with respect to this directory (unless you change the working directory from within the dockerfile).  

### docker run:
* -t : Name of image
* -i : Run an interactive window (recommended). You can respond to python input or easily stop the container with Ctrl + C
* -p \[host-port\]\[container-port\]: Map [ports](#ports).
* --mount : [Mount](#mounts) bind mounts or volumes  
* --mount type=bind,source="%cd%/mlflow_uri",target=/mlflow_uri :  
>type allows you to select a bind mount, source refers to the directory on your system, target refers to the directory in the container. %cd% is the current directory (on Windows CMD - use $PATH in Linux or Windows Powershell). Relative directories do **not** work here.

#### Mounts
A bind mount lets you modify files on your system.  
Volumes work with different docker containers at once, but you can't let non-docker programs interact with them, and they're stored in C:\ProgramData\docker\volumes.  

Here we give our container access to the mlflow URI. This way, even when starting a new container or a new image, we can still access our ML models. (And we can recover saved models within our system if we need to - though you'll need to manually change some names or copy the model file.)

#### Ports
*What is localhost?*  
It lets us host webpages using our laptop. These can't be accessed elsewhere. http://localhost:5000 is the same as http://127.0.0.1:5000. Both Flask and MLflow let us use view a UI on these ports.

Docker containers normally can't use system ports. That's one of the reasons we use virtual machines like docker (no unwanted permissions).  
However, -p or --publishs lets us give docker access to certain system ports. -p 1000:80 means that if we publish to port 80 from within our container (eg. using flask's `app.run(host="0.0.0.0", port=80)`), we can view on http://127.0.0.1:1000.  

For running in a docker container, we use host 0.0.0.0 because this publishes on all ports. On our system, we can still use host 127.0.0.1. Note that *viewing* is always done on 127.0.0.1.
# MLflow
### mlflow ui:
This command will launch the MLflow UI. It will also prevent you from using the active terminal.
```
mlflow ui --backend-store-uri <directory>
```
* --backend-store-uri : Path to mlflow_uri directory
* --host : Use 0.0.0.0 to run on all ports  
Note: When run, mlflow will output "running on http://0.0.0.0:5000" in the terminal but will actually run on http://127.0.0.1:5000