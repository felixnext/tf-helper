import numpy as np

class CyclicPolicy():
  def __init__(self, base_lr, max_lr, step_size, decay=None):
    '''constructor.

    Args:
      base_lr (float): minimal learning rate to apply
      max_lr (float): maximum learning rate to apply
      step_size (int): number of steps for one cycle (i.e. from `base_lr` to `max_lr`)
      decay (fct): optional decay function to change the values for `base_lr` and `max_lr` over time with signatue [`(lr, total_steps, epochs) => lr`]
    '''
    self.base_lr = base_lr
    self.max_lr = max_lr
    self.step_size = step_size
    self.decay = decay

  def clr_triangular(self, steps, tsteps, epochs):
    '''Defines a cyclic learning rate policy based on a triangular function.

    Example:
      `lrate = BatchLearningRateScheduler(cyc_pol.clr_triangle)`

    Args:
      steps (int): the number of steps in the current epoch
      tsteps (int): the total number of steps since training started
      epochs (int): the number of epochs since training started
    '''
    # check if learning rates should be adjusted
    max_lr = self.max_lr
    base_lr = self.base_lr
    if decay is not None:
      max_lr = self.decay(self.max_lr, tsteps, epochs)
      base_lr = self.decay(self.base_lr, tsteps, epochs)

    # calculate current position in the cycle and learning rate
    cycle = np.floor(1+tsteps/(2*self.step_size))
    x = np.abs(tsteps/self.step_size - 2*cycle + 1)
    lr = base_lr + (max_lr-base_lr)*np.maximum(0, (1-x))
    return lr

  def clr_polynomial(self, steps, tsteps, epochs):
    # check if learning rates should be adjusted
    if decay is not None:
      self.max_lr = self.decay(self.max_lr, tsteps, epochs)
      self.base_lr = self.decay(self.max_lr, tsteps, epochs)

    # TODO: implement
    return max_lr

class StepDecayPolicy():
  def __init__(self, initial_lr=0.1, drop=0.2, epochs_drop=1):
    '''constructor.

    Args:
      initial_lr (float): Inital learning rate to start with
      drop (float): the value to drop at each step (how this value is applied depends on the function used)
      epochs_drop (int): number of epochs between single drops
    '''
    self.init_lr = initial_lr
    self.drop = drop
    self.epochs_drop = epochs_drop

  def exp_step_decay(self, epoch):
      '''Defines a step decay for the learning rate based on a exponential function

      Example:
        `lrate = LearningRateScheduler(sp_pol.exp_step_decay)`

      Args:
        epoch (int): number of epochs since training started
      '''
      return self.init_lr + np.exp(epoch / 10) * self.drop

  def linear_step_decay(self, epoch):
    '''Defines a cyclic learning rate policy based on a triangular function.

    Example:
      `lrate = LearningRateScheduler(sp_pol.exp_step_decay)`

    Args:
      epoch (int): number of epochs since training started
    '''
    return self.init_lr * np.power(self.drop, np.floor((1+epoch)/self.epochs_drop))
