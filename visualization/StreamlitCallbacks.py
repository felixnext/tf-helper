
import streamlit as st
import tensorflow as tf
import math
import pandas as pd
import time


class StreamlitCallback(tf.keras.callbacks.Callback):
  '''Callback that will display data into

  Copied and modified from streamlit MNist example.

  Args:

  '''
  def __init__(self):
    pass

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
    self._epoch_chart = st.line_chart()
    self._epoch_progress = st.info("No stats yet.")
    self._epoch_summary = st.empty()

  def on_batch_end(self, batch, logs=None):
    # add data point to the graph (in 10 batch steps)
    rows = pd.DataFrame(
        [[logs["loss"], logs["accuracy"]]], columns=["loss", "accuracy"]
    )
    if batch % 10 == 0:
      self._epoch_chart.add_rows(
          {"loss": [logs["loss"]], "accuracy": [logs["accuracy"]]}
      )
    if batch % 100 == 99:
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
    # st.write('**Summary**')
    # call external test function
    self.test_step()
    # print the summary
    summary = "\n".join(
        "%(k)8s : %(v)8.5f" % {"k": k, "v": v} for (k, v) in logs.items()
    )
    st.text(summary)
    self._summary_stats.text(
        "%(epoch)8s :  %(epoch)s\n%(summary)s"
        % {"epoch": epoch, "summary": summary}
    )

  def test_step(self):
    raise NotImplementedError("Not yet implemented")
