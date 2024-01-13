# Identify whether a pizza is shown in the image
## Objective

This repository contains the final project for the [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) course provided by [DataTalks.Club](https://datatalks.club/).

The goal of the project is to apply what we have learned during the course. This project aims to develop an exemplary Kubernetes cluster deployment using [Kind](https://kind.sigs.k8s.io/docs/user/quick-start/) for [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) architecture. The binary image classifier attempts to determine whether an image shows a pizza. The overview of the approach is explained in following [video](https://github.com/DataTalksClub/machine-learning-zoomcamp/blob/master/11-kserve/05-tensorflow-kserve.md).

In summary, the cluster consists of two services. One is responsible for serving a model, while the other handles traffic maintenance. The latter is referred to as a LoadBalancer, which acts as a gateway for requests sent to the cluster. Requests are registered and forwarded thanks to port forwarding between architectural components.

The following video shows how the project works in a humorous way
(click the image below to start short youtube video):

[![Video Title](https://i.ytimg.com/vi/tWwCK95X6go/hq720.jpg?sqp=-oaymwEcCNAFEJQDSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLBndyO3_OWhyfNPMm7hMGzV7jX2fw)](https://youtu.be/tWwCK95X6go?si=LPqjv3k_NyPgqaAq)

*Image from Silicon Valley TV show, created by Mike Judge, John Altschuler, and Dave Krinsky.*

## Dataset

The dataset used to train the image classifier can be found on [Kaggle](https://www.kaggle.com/datasets/carlosrunner/pizza-not-pizza). You can download the dataset directly from there or explore it via [Kaggle](https://www.kaggle.com/) itself.

Due to the nature of the problem - a binary image classifier - conducted exploratory data analysis (EDA) was very elementary. The dataset consists of 983 pizza images and 983 non-pizza images. Please refer to the notebook titled: [01_eda](https://github.com/KonuTech/machine-learning-zoomcamp-capstone-02/blob/main/notebooks/01_eda.ipynb).

## Training

The experiments conducted resulted in several different training runs, using various architectures of CNN (convolutional neural networks). Various versions of pre-trained models were employed in the hope of improving the quality of the champion model. The impact of so-called Transfer Learning can be observed both [here](https://github.com/KonuTech/machine-learning-zoomcamp-capstone-02/blob/main/notebooks/02_get_champion_binary_classifier.ipynb) and on [Kaggle](https://www.kaggle.com/code/konutech/machine-learning-zoomcamp-pizza-classifier/notebook), where you can run the notebook responsible for training the champion model yourself.

The list of pre-trained models used in experiments where the so-called transfer learning approach was applied:
* Xception
* EfficientNetB3
* InceptionV3
* EfficientNetB5
* VGG16

## Model deployment and serving

The model was first tested as a containerized Flask app. Afterwards, the model was served as a Kind Kubernetes cluster. To see how to apply the model, look into the details below.

### Applied technologies

| Name | Scope |
| --- | --- |
| Jupyter Notebooks | EDA, experiments, scoring |
| TensorFlow | pre-processing, feature engineering, transfer learning, serving|
| Flask | web service |
| pylint | Python static code analysis |
| black | Python code formatting |
| isort | Python import sorting |
| Docker Desktop | Containerization of servicesc |
| Kind | Kubernetes cluster |

### Architecture

Here is a high-level schema of an architecture:
<img src="static/architecture.jpg" width="60%"/>

Project Structure
------------
    ├── data
    │   ├── pizza_not_pizza
    │   ├── not_pizza
    │   ├── pizza
    ├── k8s
    │   ├── pizza-model
    │   │   ├── assets
    │   │   ├── variables
    ├── models
    ├── notebooks
    │   ├── training_logs
    │   │   ├── pizza_classification
    │   │   │   │   ├── 20240106-175439
    │   │   │   │   ├── train
    │   │   │   │   ├── validation
    |   │   │   │   ├── 20240106-181636
    │   │   │   │   ├── train
    │   │   │   │   ├── validation
    |
    |
    |
    ├── scoring
    │   ├── logs
    │   ├── models

## Reproducibility

##### Pre-requisties

* python 3.9 or above
* Docker Desktop, Kind, kubectl
* pip3, pipenv
* git-lfs

### Docker deployment (containerization)
Before deploying to a Kubernetes cluster, we will test the model and the gateway locally using Docker Compose. To do this, we will utilize the [Dockerfile](https://github.com/KonuTech/machine-learning-zoomcamp-capstone-02/blob/main/scoring/Dockerfile), which defines the service. Additionally, a script named [predict_test.py](https://github.com/KonuTech/machine-learning-zoomcamp-capstone-02/blob/main/scoring/predict_test.py) is defined for testing the model predictions. The above script is stored locally. It passes URL to [predict.py]() which is stored on a container along with a copy of a TensorFlow model.

Once in ./scoring directory - where [Dockerfile](https://github.com/KonuTech/machine-learning-zoomcamp-capstone-02/blob/main/scoring/Dockerfile) is present - you can build up the image with:
```
$ docker build -t machine-learning-zoomcamp-capstone-02 .
[+] Building 120.4s (13/13) FINISHED                                                                                                                                                                                docker:default 
 => [internal] load .dockerignore                                                                                                                                                                                             0.1s 
 => => transferring context: 2B                                                                                                                                                                                               0.0s 
 => [internal] load build definition from Dockerfile                                                                                                                                                                          0.1s 
 => => transferring dockerfile: 389B                                                                                                                                                                                          0.0s 
 => [internal] load metadata for docker.io/library/python:3.9.3-slim                                                                                                                                                          2.4s 
 => [1/8] FROM docker.io/library/python:3.9.3-slim@sha256:3edfa765f8f77f333c50222b14552d0d0fa9f46659c1ead5f4fd10bf96178d3e                                                                                                    0.0s 
 => [internal] load build context                                                                                                                                                                                             0.0s 
 => => transferring context: 2.98kB                                                                                                                                                                                           0.0s 
 => CACHED [2/8] RUN pip install pipenv                                                                                                                                                                                       0.0s 
 => CACHED [3/8] WORKDIR /app                                                                                                                                                                                                 0.0s 
 => [4/8] COPY [predict.py, Pipfile, Pipfile.lock, ./]                                                                                                                                                                        0.1s 
 => [5/8] RUN pipenv install --system --deploy                                                                                                                                                                              106.8s 
 => [6/8] RUN mkdir models                                                                                                                                                                                                    0.6s
 => [7/8] RUN mkdir logs                                                                                                                                                                                                      0.7s
 => [8/8] COPY [models/model_xception_2024-01-06_22-44-37.keras, models/]                                                                                                                                                     0.8s
 => exporting to image                                                                                                                                                                                                        8.6s
 => => exporting layers                                                                                                                                                                                                       8.6s
 => => writing image sha256:647e632aaf90f351add90a802284d99447063eaa62e454df07223aa86f21d60e                                                                                                                                  0.0s
 => => naming to docker.io/library/machine-learning-zoomcamp-capstone-02                                                                                                                                                      0.0s
```
After successful build of image you can run the Flask app stored iside it:
```
$ docker run -it --rm -p 6969:6969 --entrypoint=bash machine-learning-zoomcamp-capstone-02
root@0da539eb1ed1:/app#

root@5caec8e9b5fb:/app# ls
Pipfile  Pipfile.lock  logs  models  predict.py

root@0da539eb1ed1:/app# python predict.py
2024-01-13 21:31:57.584757: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2024-01-13 21:31:57.584810: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2024-01-13 21:31:58.884489: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2024-01-13 21:31:58.884542: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2024-01-13 21:31:58.884593: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (0da539eb1ed1): /proc/driver/nvidia/version does not exist
2024-01-13 21:31:58.884803: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
 * Serving Flask app 'predict'
 * Debug mode: on
2024-01-13 21:32:00.626125: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2024-01-13 21:32:00.626184: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2024-01-13 21:32:01.796997: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
2024-01-13 21:32:01.797050: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
2024-01-13 21:32:01.797097: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (0da539eb1ed1): /proc/driver/nvidia/version does not exist
2024-01-13 21:32:01.797293: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
```
#### Testing with python script
Now from local terminal we can run:
```
KonuTech@DESKTOP-D7SFLUT MINGW64 ~/machine-learning-zoomcamp-capstone-02 (main)
$ python scoring/predict_test.py
```
To see that containerized prediction app returend:
```
2024-01-13 21:32:01.797293: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
It is a pizza!
```
<img src="static/it_is_a_pizza_docker.jpg" width="80%"/>

##### Dependencies
The list of dependencies for a deployment using Kind is available [here](https://github.com/KonuTech/machine-learning-zoomcamp-capstone-02/blob/main/scoring/Pipfile).
As always first do the following:
```
pip install pipenv
pipenv shell
```
Next, install dependencies listed under Pipfile using following command:
```
pipenv install
```
### Kind deployment (Kubernetes)

##### Dependencies
The list of dependencies for successful deployment of Kind cluster locally is available [here](https://github.com/KonuTech/machine-learning-zoomcamp-capstone-02/blob/main/k8s/Pipfile).
First, as always do the following:
```
pip install pipenv
pipenv shell
```
Next, install dependencies listed under Pipfile using following command:
```
pipenv install
```
We are going to use the tool [Kind](https://kind.sigs.k8s.io/docs/user/quick-start/) to create a Kubernetes cluster locally. A single pod is created for each of the services: one pod for a model-serving service and another pod for the creation of a so-called gateway. This is illustrated in the architecture schema image shown previously.

Here, I am assuming that you were aleady able to instal [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/). 

#### Prerequisites

If you are a Windows user you can download Kind using following URL:

```
curl.exe -Lo kind-windows-amd64.exe https://kind.sigs.k8s.io/dl/v0.20.0/kind-windows-amd64
Move-Item .\kind-windows-amd64.exe c:\kind\kind.exe
```

Next you can create a cluster with:
```
PS C:\kind> .\kind.exe create cluster
```
You should see:
```
Creating cluster "kind" ...
 • Ensuring node image (kindest/node:v1.27.3) 🖼  ...
 ✓ Ensuring node image (kindest/node:v1.27.3) 🖼
 • Preparing nodes 📦   ...
 ✓ Preparing nodes 📦
 • Writing configuration 📜  ...
 ✓ Writing configuration 📜
 • Starting control-plane 🕹️  ...
 ✓ Starting control-plane 🕹️
 • Installing CNI 🔌  ...
 ✓ Installing CNI 🔌
 • Installing StorageClass 💾  ...
 ✓ Installing StorageClass 💾
Set kubectl context to "kind-kind"
You can now use your cluster with:

kubectl cluster-info --context kind-kind

Thanks for using kind! 😊
```
Check if cluster is up and running with:
```
PS C:\kind> kubectl cluster-info --context kind-kind
```
You should see something like following:
```
Kubernetes control plane is running at https://127.0.0.1:59542
CoreDNS is running at https://127.0.0.1:59542/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

To further debug and diagnose cluster problems, use 'kubectl cluster-info dump'.
```
Now you can create bunch of Docker images:
```
docker build -t machine-learning-zoomcamp-capstone-02:xception-001 \
    -f pizza-model.dockerfile .

docker build -t machine-learning-zoomcamp-capstone-02-gateway:001 \
    -f pizza-gateway.dockerfile .
```
Next, load previously created and tested Docker images into the cluster:
```
C:\kind>kind load docker-image machine-learning-zoomcamp-capstone-02:xception-001
Image: "machine-learning-zoomcamp-capstone-02:xception-001" with ID "sha256:5e45971598ba189a7bd5f36a182a2e27272303a35a498cfa0a2574ba357e8ffd" not yet present on node "kind-control-plane", loading...

C:\kind>.\kind.exe load docker-image machine-learning-zoomcamp-capstone-02-gateway:001
Image: "machine-learning-zoomcamp-capstone-02-gateway:001" with ID "sha256:8168d041ad2e8d9f0c227fd5b9b56e1db4236c6e8766cc094d086866fa66e480" not yet present on node "kind-control-plane", loading...
```
Now, we can create resources from .yaml files:
```
$ kubectl apply -f model-deployment.yaml
deployment.apps/tf-serving-pizza-model created

$ kubectl apply -f gateway-deployment.yaml
deployment.apps/gateway created

$ kubectl apply -f model-service.yaml
service/tf-serving-pizza-model created

kubectl apply -f gateway-service.yaml
deployment.apps/gateway created
```
We can check any running pod or service:
```
$ kubectl get pod
NAME                                     READY   STATUS    RESTARTS        AGE
gateway-549c6cb9bc-bszf8                 1/1     Running   5 (3h14m ago)   3d23h
tf-serving-pizza-model-c956959f9-rdhqv   1/1     Running   5 (3h14m ago)   4d1h

$ kubectl get services
NAME                     TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)        AGE
gateway                  LoadBalancer   10.96.189.170   <pending>     80:30322/TCP   3d23h
kubernetes               ClusterIP      10.96.0.1       <none>        443/TCP        40d
tf-serving-pizza-model   ClusterIP      10.96.115.145   <none>        8500/TCP       4d1h
```
The last thing to do is to forward ports:
```
kubectl port-forward tf-serving-pizza-model-c956959f9-rdhqv 8500:8500
kubectl port-forward gateway-549c6cb9bc-bszf8 9696:9696
kubectl port-forward service/gateway 8080:80
```

#### Testing with python script
Now, since the ports were forwared we can try to make a prediction:
```
python k8s/predict_test.py
```
After that we can confirm if the prediction was done thanks to the log from the gateway:
```
$ kubectl logs gateway-549c6cb9bc-bszf8
[2024-01-13 08:59:42 +0000] [1] [INFO] Starting gunicorn 21.2.0
[2024-01-13 08:59:42 +0000] [1] [INFO] Listening at: http://0.0.0.0:9696 (1)
[2024-01-13 08:59:42 +0000] [1] [INFO] Using worker: sync
[2024-01-13 08:59:42 +0000] [10] [INFO] Booting worker with pid: 10
2024-01-13 08:59:44.013754: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2024-01-13 08:59:44.013866: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2024-01-13 12:15:29,210 - DEBUG - Received URL for prediction: https://m.kafeteria.pl/shutterstock-84904912-9cb8cae338,730,0,0,0.jpg
2024-01-13 12:15:29,482 - DEBUG - Sending prediction request to TensorFlow Serving.
2024-01-13 12:15:30,713 - DEBUG - Received prediction response from TensorFlow Serving.
2024-01-13 12:15:30,713 - DEBUG - Prediction result: It's a pizza!
```
<img src="static/it_is_a_pizza_cluster.jpg" width="80%"/>


### Peer review criterias - a self assassment:
* Problem description
    * 2 points: The problem is well described and it's clear what the problem the project solves
* EDA
    * 2 points: Extensive EDA (ranges of values, missing values, analysis of target variable, feature importance analysis).
    For images: analyzing the content of the images. For texts: frequent words, word clouds, etc
* Model training
    * 3 points: Trained multiple models and tuned their parameters. For neural networks: same as previous, but also with tuning: adjusting learning rate, dropout rate, size of the inner layer, etc
* Exporting notebook to script
    * 1 point: The logic for training the model is exported to a separate script
* Reproducibility
    * 1 point: It's possible to re-execute the notebook and the training script without errors. The dataset is committed in the project repository or there are clear instructions on how to download the dat 
* Model deployment
    * 2 points: Model is deployed (with Flask, BentoML or a similar framework)
* Dependency and enviroment managemen
    * 2 points: Provided a file with dependencies and used virtual environment. README says how to install the dependencies and how to activate the env.
* Containerization
    * 2 points: The application is containerized and the README describes how to build a contained and how to run it
* Cloud deployment
    * 2 points: There's code for deployment to cloud or kubernetes cluster (local or remote). There's a URL for testing - or video/screenshot of testing it