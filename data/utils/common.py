

class DataType(Enum):
    '''Types of possible input data.'''
    TRAINING    = 0
    DEVELOPMENT = 1
    TESTING     = 2

class ImageSettings():
  '''Defines image settings for the data to load.'''

  def __init__(self, resize, fill, pad, pad_color=(0,0,0)):
    pass

  def transform(img):
    '''Transforms the given image according to the specifications.'''
    return img


class ResizeMode(Enum):
    '''Types of resize functions.'''
    FIT         = 0
    STRETCH     = 1
    PAD_COLOR   = 2
    PAD_MEAN    = 3
    PAD_EDGE    = 4
    PAD_RANDOM  = 5

class FillMode(Enum):
    '''Fill Mode for image patches.'''
    COLOR  = 0
    MEAN   = 1
    RANDOM = 2

class PadMode(Enum):
    '''Defines the type of padding mode.'''
    EDGE    = 0
    CENTER  = 1
