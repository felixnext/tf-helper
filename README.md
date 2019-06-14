# tf-helper

This is a collection of helper function that I found useful during my work with TensorFlow and Keras.

Feel free to make pull requests if you see potential improvements.


## Docs

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

However, there are also some lower level function
