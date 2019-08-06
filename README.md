# YELP Dataset Analysis:

- Exploratory Data Analysis (EDA)
- Natural Language Processing (NLP)
- Image Classification

# File descriptions

I provided jupyter notebooks for easy data exploring, visualization and interactive evaluation.

- **YELP_Dataset_EDA.ipynb**: Explanatory data analysis (EDA) and AFINN-based text processing parts.

- **YELP_Dataset_NLP.ipynb**: Natural language processing (NLP) for text classification and embedding visualization. 

- **YELP_Dataset_IRC.ipynb**: Image classification with convolutional neural networks (CNN) and transfer learning.

- **functions.py**: auxiliary functions used on the jupyter notebooks.

- **Dockerfile**: file to create a Docker image.

# Getting started

1- Download all the files (3 jupyter notebooks, 1 python file and 1 Dockerfile) and keep them in the same location.

2- Execute the Dockerfile to create an image from the base docker image jupyter/datascience-notebook. All files will be copied on the folder "/home/jovyan/work" inside the docker image.

3- Download the data files yelp_dataset.tar.gz and yelp_photos.tar.gz from https://www.yelp.com/dataset_challenge/dataset.

4- Copy, and then unzip those files into the same folder "/home/jovyan/work" inside the docker image.

5- Execute the command jupyter notebook on the terminal to open the jupyter notebooks above.
