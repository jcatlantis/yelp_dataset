FROM jupyter/datascience-notebook
MAINTAINER JuanKrlos jc.atlantis@gmail.com

RUN pip install -U pip

# installing and updating required packages

RUN pip install afinn
RUN pip install tensorflow==2.0.0-beta1
RUN pip install opencv-python
RUN conda update --all

# due to compatibiltiy issues

RUN pip install numpy==1.16.4

WORKDIR /home/jovyan/work
ADD . /home/jovyan/work