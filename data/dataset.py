'''
Base dataset class used to load various types of datasets.

Note that the datasets are loaded through generators in a pandas dataframe (for quick selection of arguments)
'''

from loader import classification as clsfy


class ImageDataset():
  '''Class that stores an image dataset either for localization or classification.'''

  def __init__(self, gen, classes):
    self._classes = classes

  def index(self):
    '''Retrieves the index of the dataset.'''
    pass

  def classes(self):
    '''Returns a list of the available classes for the dataset.'''
    return self._classes

  def loadKitti(folder):
    '''Loads kitti dataset.'''

  def loadClasses(folder, classes=None, settings=None, only_set=None):
    '''Loads a classification dataset.

    The dataset has to be stored according to predefined format. This means it should either

    Args:
      folder (str): path to the dataset
      classes (list): predefined list of classes that should be loaded
      settings (ImageSettings): settings how images should be handled for normalization
      only_set (list): Defines a list of `utils.DataType` that specify if only partial areas of the dataset (e.g. training) should be loaded

    Returns:
      ds (Dataset): Created dataset that allows to retrieve the relevant data and inject them into a keras or tensorflow pipeline
    '''
    # safty: check if the folder exists
    if not os.path.exists(folder):
      raise IOError("Specified folder ({}) does not exist!".format(folder))

    # load the relevant classes
    if classes is None:
      classes = clsfy._find_classes(folder, only_set)

    # load the dataset
    # TODO: allow to load all dataset types separatly to put them into data?
    # TODO: otherwise split data at runtime?
    gen = clsfy._gen_cls(folder, classes, shuffle, only_set, size, one_hot,
                         beard_format, show_btype, resize, pad_color, pad_mode, debug)

    return new ImageDataset(gen, classes)
