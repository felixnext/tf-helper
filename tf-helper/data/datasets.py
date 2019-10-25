
from tensorflow.keras import datasets
from tensorflow.keras import utils
from bunch import Bunch

def mnist_cnn():
  '''Loads mnist CNN data.'''
  (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

  img_width = 28
  img_height = 28

  x_train = x_train.astype("float32")
  x_train /= 255.0
  x_test = x_test.astype("float32")
  x_test /= 255.0

  # reshape input data
  x_train = x_train.reshape(x_train.shape[0], img_width, img_height, 1)
  x_test = x_test.reshape(x_test.shape[0], img_width, img_height, 1)

  # one hot encode outputs
  y_train = utils.to_categorical(y_train)
  y_test = utils.to_categorical(y_test)
  num_classes = y_test.shape[1]

  # create bunch package
  return Bunch({'test': Bunch({'x': x_test, 'y': y_test}), 'train': Bunch({'x': x_train, 'y': y_train}), 'size': (img_width, img_height, 1), 'num_classes': num_classes, 'class_names': None})

def cifar10():
  '''Load Cifar10 data.'''
  (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

  train_images, test_images = train_images / 255.0, test_images / 255.0

  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  num_classes = len(class_names)

  train_labels = utils.to_categorical(train_labels, num_classes)
  test_labels = utils.to_categorical(test_labels, num_classes)

  return Bunch({'test': Bunch({'x': test_images, 'y': test_labels}), 'train': Bunch({'x': train_images, 'y': train_labels}), 'size': (32, 32, 3), 'num_classes': num_classes, 'class_names': class_names})

def places365():
  '''Load small Places 365 Dataset.'''
  (train_images, train_labels), (test_images, test_labels) = datasets.places365_small.load_data()

  train_images, test_images = train_images / 255.0, test_images / 255.0

  class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  num_classes = len(class_names)

  train_labels = utils.to_categorical(train_labels, num_classes)
  test_labels = utils.to_categorical(test_labels, num_classes)
