from tensorflow.keras import backend
from tensorflow.keras.callbacks import Callback

class BatchLearningRateScheduler(Callback):
  '''Learning Rate Scheduler that can change the learning rate on a batch level.'''

  def __init__(self, schedule, verbose=0):
    '''Constructur.

    The schedule function takes 3 parameters:
      steps (int): number of batches processed in the current epoch
      total_steps (int): number of batches processed since the beginning of the training
      epochs (int): total number of epochs since the beginning of the training

    Note that the new learning rate is set after each batch.

    Args:
      schedule (fct): Function that defines the policy with signature [`(steps, total_steps, epochs) => learning_rate`]
    '''
    super().__init__()
    self.schedule = schedule
    self.verbose = verbose
    self.steps = 0
    self.epochs = 0
    self.total_steps = 0

  def on_epoch_end(self, epoch, logs=None):
    self.epochs += 1
    self.steps = 0

  def on_batch_end(self, batch, logs=None):
    self.steps += 1
    self.total_steps += 1
    lr = self.schedule(self.steps, self.total_steps, self.epochs)
    backend.set_value(self.model.optimizer.lr, lr)
