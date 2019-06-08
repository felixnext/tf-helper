import tensorflow as tf

# create the metric for keras
def binary_accuracy(y, y_pred, threshold=0.5):
  '''
  Args:
    y (tf.Tensor):
    y_pred (tf.Tensor):
    threshold (tf.Tensor):

  Returns:
    Tensor with the accuracy value
  '''
    y_pred = tf.cast(tf.greater(y_pred, threshold), tf.float32)
    acc = tf.reduce_min(tf.cast(tf.equal(y, y_pred), tf.float32), axis=1)
    return tf.reduce_sum(acc) / tf.cast(tf.shape(acc)[0], tf.float32)
