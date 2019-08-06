# YELP Dataset Analysis:

- Exploratory Data Analysis (EDA)
- Natural Language Processing (NLP)
- Image Classification

# File descriptions

I provided jupyter notebooks for easy data exploring, visualization and interactive evaluation.

- File 1: Explanatory data analysis and AFINN-based text processing parts.

- File 2: Natural language processing for text classification and embedding visualization. 

- File 3: Image classification with convolutional neural networks (CNN) and transfer learning.

- functions.py: auxiliary functions used on the jupyter notebooks.

- Dockerfile: file to create a Docker image.

# Getting started

- Download all the files (3 jupyter notebooks, 1 python file and 1 Dockerfile) and keep them in the same location.

- Execute the Dockerfile to create an image from the base docker image jupyter/datascience-notebook. All files will be copied on the folder "/home/jovyan/work" inside the docker image.

- Download the data files yelp_dataset.tar.gz and yelp_photos.tar.gz from https://www.yelp.com/dataset_challenge/dataset.

- Copy and unzip those files into the same folder "/home/jovyan/work" inside the docker image.

- Execute the command jupyter notebook in the terminal to open the notebooks.
