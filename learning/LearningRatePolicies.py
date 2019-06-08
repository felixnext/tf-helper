import numpy as np

def clr_triangle(steps, tsteps, epochs):
  '''Defines a cyclic learning rate policy based on a triangular function.

  Example:
    `lrate = BatchLearningRateScheduler(clr)`

  Args:
    steps (int):
    tsteps (int):
    epochs (int):
  '''
  cycle = np.floor(1+tsteps/(2*step_size))
  x = np.abs(tsteps/step_size - 2*cycle + 1)
  lr = base_lr + (max_lr-base_lr)*np.maximum(0, (1-x))
  return lr
