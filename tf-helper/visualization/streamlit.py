'''
Various Visualization functions adapted from Streamlit examples

Source: https://github.com/streamlit/streamlit/tree/develop/examples
'''

import streamlit as st
import tensorflow as tf
import math
import pandas as pd
import time


class StreamlitCallback(tf.keras.callbacks.Callback):
  '''Callback that will display data into

  Copied and modified from streamlit MNist example.

  Example:
  ```
  class MyCallback(StreamlitCallback):
    def test_step(self):
      # ...
  ```

  Args:
  show_epoch_chart (bool) = Defines if the epoch chart should be shown
  epoch_update (int): Batch-Frequency to update the epoch chart
  summary_update (int): Batch-Frequency to update the overall chart
  test_imgs (np.array): List of images to test the model against (if `None` no test is performed)
  test_count (int): Number of random selected images to test against
  test_names (list): List of labels to use for the prediction output
  '''
  def __init__(self, show_epoch_chart=True, epoch_update=10, summary_update=100, test_imgs=None, test_count=6, test_names=None):
    self._show_epoch = show_epoch_chart
    self._ep_freq = epoch_update
    self._sum_freq = summary_update
    self._img_test = test_imgs
    self._img_test_size = test_count
    self._img_test_names = test_names

  def on_train_begin(self, logs=None):
    # Run on beginning of epoch
    st.header("Summary")
    # create chart handles for later usage
    self._summary_chart = st.area_chart()
    self._summary_stats = st.text("%8s :  0" % "epoch")
    st.header("Training Log")

  def on_epoch_begin(self, epoch, logs=None):
    self._ts = time.time()
    self._epoch = epoch
    # create new section
    st.subheader("Epoch %s" % epoch)
    # create linechart for data in current epoch
    if self._show_epoch == True:
      self._epoch_chart = st.line_chart()
    self._epoch_progress = st.info("No stats yet.")
    self._epoch_summary = st.empty()

  def on_batch_end(self, batch, logs=None):
    # add data point to the graph (in 10 batch steps)
    rows = pd.DataFrame(
        [[logs["loss"], logs["accuracy"]]], columns=["loss", "accuracy"]
    )

    # update epoch chart
    if batch % self._ep_freq == 0 and self._show_epoch == True:
      self._epoch_chart.add_rows(
          {"loss": [logs["loss"]], "accuracy": [logs["accuracy"]]}
      )

    # update summary chart
    if batch % self._sum_freq == 0:
      self._summary_chart.add_rows(rows)

    # update texts
    percent_complete = logs["batch"] * logs["size"] / self.params["samples"]
    self._epoch_progress.progress(math.ceil(percent_complete * 100))
    ts = time.time() - self._ts
    self._epoch_summary.text(
        "loss: %(loss)7.5f | accuracy: %(accuracy)7.5f | ts: %(ts)d"
        % {"loss": logs["loss"], "accuracy": logs["accuracy"], "ts": ts}
    )

  def on_epoch_end(self, epoch, logs=None):
    # call external test function
    self.custom_epoch_end(epoch, logs)

    # test images
    if self._img_test is not None:
      self._predict_imgs()

    # print the summary
    summary = "\n".join(
        "%(k)8s : %(v)8.5f" % {"k": k, "v": v} for (k, v) in logs.items()
    )
    st.text(summary)
    self._summary_stats.text(
        "%(epoch)8s :  %(epoch)s\n%(summary)s"
        % {"epoch": epoch, "summary": summary}
    )

  def _predict_imgs(self):
    '''Predict data for the images.'''
    indices = np.random.choice(len(self._img_test), self._img_test_size)
    test_data = self._img_test[indices]
    prediction = np.argmax(self.model.predict(test_data), axis=1)
    # update predictions
    if self._img_test_names is not None:
      prediction = np.array(self._img_test_names)[prediction.astype('int')]

    # display as images
    st.image(1.0 - test_data, caption=prediction)

  def custom_epoch_end(self, epoch, logs=None):
    '''Abstract Function to implement inference and introspection functions.'''
    pass
