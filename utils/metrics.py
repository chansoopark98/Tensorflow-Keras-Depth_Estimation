import tensorflow as tf


def depth_inverse(depth, max_depth=1000.0):
    depth = max_depth / depth
    depth = tf.where(tf.math.is_inf(depth), 0., depth)
    return depth

def ssim_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = depth_inverse(depth=y_true)
    y_pred = depth_inverse(depth=y_pred)
    ssim = tf.image.ssim(y_true, y_pred, 1000.0)
    return ssim

class RMSE(tf.keras.metrics.RootMeanSquaredError):
  def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = depth_inverse(y_true)
        y_pred = depth_inverse(y_pred)

        return super().update_state(y_true, y_pred, sample_weight)