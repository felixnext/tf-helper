from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend
import matplotlib.pyplot as plt
try:
  from IPython.display import clear_output
except ImportError:
  pass

# create a callback to plot the losses
class PlotLossesCallback(Callback):
  '''Callback to plot losses against.'''

  def __init__(self, plot_type, plot_lr=False, draw_every_batch=False, clear_ipython=False, use_ggplot=True):
    '''Creates new instance of the callback.

    Args:
      plot_type (str): Defines the type of plot (options: ['epoch', 'batch', 'lr', 'lr_batch'])
      plot_lr (bool): Defines if the learning rate should be plotted (only applies to `'epoch'` and `'batch'`)
      clear_ipython (bool): Defines if ipython should be cleared after the
    '''
    plot_type = plot_type.lower()
    self.plot_type = plot_type if ['epoch', 'batch', 'lr', 'lr_batch'].index(plot_type) >= 0 else "epoch"
    self.clear_ipython = clear_ipython
    self.plot_lr = plot_lr
    self.draw_every_batch = draw_every_batch

    # change plot layout
    if use_ggplot:
      plt.style.use('ggplot')

  def on_train_begin(self, logs={}):
    self.count_epoch = 0
    self.count_batch = 0
    self.x = []
    self.x_val = []
    self.losses = []
    self.val_losses = []
    self.lr = []

    self.fig = plt.figure()

    self.logs = []

  def _get_value(self):
    '''Retrieve the current value based on the type of plot to use.'''
    if self.plot_type == 'batch':
      return self.count_batch
    elif self.plot_type in ['lr_batch', 'lr']:
      return backend.eval(self.model.optimizer.lr)
    elif self.plot_type == 'epoch':
      return self.count_epoch

  def on_batch_end(self, batch, logs={}):
    # check for plotting
    if self.plot_type in ['batch', 'lr_batch']:
      # update the relevant values
      self.x.append(self._get_value())
      self.losses.append(logs.get('loss'))
      if self.plot_lr:
        self.lr.append(backend.eval(self.model.optimizer.lr))

      # plot the relevant data (if enabled)
      if self.draw_every_batch:
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x_val, self.val_losses, label="val_loss")
        # check if learning rate should be plotted
        if self.plot_type == 'batch' and self.plot_lr:
          plt.plot(self.x, self.lr, label="lr")

        # store the relevant plot data
        plt.legend()
        plt.show();

    # update the counter
    self.count_batch += 1

  def on_epoch_end(self, epoch, logs={}):
    self.logs.append(logs)
    if self.plot_type in ['batch', 'lr_batch']:
      self.x_val.append(self._get_value())
    else:
      self.x.append(self._get_value())
      self.x_val.append(self._get_value())
      self.losses.append(logs.get('loss'))
      if self.plot_lr:
        self.lr.append(backend.eval(self.model.optimizer.lr))
    self.val_losses.append(logs.get('val_loss'))

    # update the counter
    self.count_epoch += 1

    # check if ipython output should be cleared
    if self.clear_ipython:
      try:
        clear_output(wait=True)
      except:
        pass

    # plot the relevant data
    plt.plot(self.x, self.losses, label="loss")
    plt.plot(self.x_val, self.val_losses, label="val_loss")
    if self.plot_type in ['batch', 'epoch'] and self.plot_lr:
      plt.plot(self.x, self.lr, label="lr")

    # store the relevant plot data
    plt.legend()
    plt.show();
