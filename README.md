# tf-helper

This is a collection of helper function that I found useful during my work with TensorFlow and Keras.

## Getting Started

The repo is structured into different categories (e.g. `data` or `learning`). Each category contains a number of python files that are designed to be copied out of the project and used in various projects.

The files came into being during a variety of projects I did with TF and keras and are basically code snippets that I have found myself reusing quite often.

## Docs

This section contains a basic overview of the different categories and how the scripts can be used.

### Learning

This is for various learning policies and schedulers.

### Metrics

Various metrics that are not defined by default in tensorflow/keras.

### Visualization

Various functions to visualize learning processes for easier introspection.

### Data

The `data` package is currently mainly focused on loading (and storing) data for image processing. This includes classification datasets as well as detection datasets in the kitti format.

Most functions are exposed through the `ImageDataset` class, which allows to inject the functions directly into the keras and TF dataset APIs.

Example:
```python
# Loading of classification datasets
from data import ImageDataset
from data.utils import ImageSettings

# define the image settings (to generate standardized images)
imgSettings = ImageSettings()
# load the dataset
ds = ImageDataset.loadClasses("/dataset/folder", settings=imgSettings)

# inject the dataset into keras
# TODO
```

However, there are also some lower level function to manipulate the image data directly (TODO)

## ToDo List

* [ ] Add Jupyter Notebooks to demonstrate usage of individual scripts on dummy data.
* [ ] Update Documentation on the scripts
* [ ] Complete implementation ofdata functions
