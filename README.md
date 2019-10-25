# tf-helper

This is a collection of helper function that I found useful during my work with TensorFlow and Keras.

TODO: Note about purpose and integration of different code bodies.

## Getting Started

You can install the library by using pip:

```bash
pip install .
```

After that simply import the library and use according to docs (see below):

```python
import tf_helper as tfh
```

> Note: The library was build and tested against tensorflow-2.0

## Docs

The library is split into different sub-modules, based on their purpose:

* `data` - all functions relevant to data loading and augmentation (including default datasets)

This section contains a basic overview of the different categories and how the scripts can be used.

### Learning

This is for various learning policies and schedulers.

### Metrics

Various metrics that are not defined by default in tensorflow/keras.

### Visualization

Various functions to visualize learning processes for easier introspection.

**Streamlit Callback**

This callback shows the general progress of the learning (together with loss data). It also allows to implement the `test_step` function to provide
additional information (e.g. inference tests):

Example:
```python
from visualization.StreamlitCallback import StreamlitCallback

class MyCallback(StreamlitCallback):
  def __init__(self, x_test, class_names=None):
    self._x_test = x_test
    self._names = class_names

  def test_step(self):
    # st.write('**Summary**')
    indices = np.random.choice(len(self._x_test), 36)
    test_data = self._x_test[indices]
    prediction = np.argmax(self.model.predict(test_data), axis=1)
    # update predictions
    if self._names is not None:
      prediction = np.array(self._names)[prediction.astype('int')]

    # display as images
    st.image(1.0 - test_data, caption=prediction)
```

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

## Sources

List of research and sources relevant to this repository:

* StreamlitCallback: [Streamlit](https://github.com/streamlit/streamlit/tree/develop/examples)
* Cyclic Learning Rate: [bckenstler](https://github.com/bckenstler/CLR/blob/master/clr_callback.py)
* Look Ahead Optimizer: [CyberZHG](https://github.com/CyberZHG/keras-lookahead)
* RAdam: [CyberZHG](https://github.com/CyberZHG/keras-radam)

> Note: In most cases I have copied and adjusted the code. Notes for that are in the headers of the respective files.

## ToDo List

* [ ] Add Jupyter Notebooks to demonstrate usage of individual scripts on dummy data.
* [ ] Update Documentation on the scripts
* [ ] Complete implementation ofdata functions

## Research to Implement

* [ ] Single neuron visualization (e.g. lucid)
* [ ] Look Ahead and RAdam Learning Functions

## Known Problems

* Streamlit uses Vega, which can run into problems with Python 3.5.2
