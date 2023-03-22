import tensorflow as tf


def depth_inverse(depth, max_depth=1.0):
    depth = max_depth / depth
    # depth = tf.where(tf.math.is_inf(depth), 0., depth)
    return depth

def ssim_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    ssim = tf.image.ssim(y_true, y_pred, 1./0.05)
    return ssim

def delta_threshold_metric(y_true, y_pred):
    # Calculate the ratio of the predicted depth to the ground truth depth and vice versa
    ratio_1 = y_pred / y_true
    ratio_2 = y_true / y_pred

    # Compute the maximum of the two ratios for each pixel
    max_ratios = tf.math.maximum(ratio_1, ratio_2)

    # Calculate the percentage of pixels that fall within the threshold
    within_threshold = tf.math.less_equal(max_ratios, 1.25)
    metric = tf.math.reduce_mean(tf.cast(within_threshold, tf.float32))

    return metric

class RMSE(tf.keras.metrics.RootMeanSquaredError):
  def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # y_true = depth_inverse(y_true)
        # y_pred = depth_inverse(y_pred)

        return super().update_state(y_true, y_pred, sample_weight)