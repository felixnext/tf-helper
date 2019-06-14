'''Defines handling of IO folder structures.'''

import os, glob
from common import DataType

def dict_folders():
  '''Returns dict with all folder combinations.'''
  return {
    DataType.TRAINING: ["train", "training"],
    DataType.DEVELOPMENT: ["val", "validation", "valid", "dev", "develop", "development"],
    DataType.TESTING: ["test", "testing"]
}

def only_folders(only):
  # generate data
  folders = dict_folders()

  # filter the data if relevant
  if only is not None:
      if isinstance(only, DataType): only = [only]
      folders = {k: v for k, v in folders.items() if k in only}
  return folders
