import tensorflow as tf

def ssim_metric(y_true, y_pred):
    ssim = tf.image.ssim(y_true * 100, y_pred * 100, 100.0)
    return ssim



class RMSE(tf.keras.metrics.RootMeanSquaredError):
  def update_state(self, y_true, y_pred, sample_weight=None):
        # depth = 10. / depth
        # depth = tf.where(tf.math.is_inf(depth), 0., depth)
        # depth = tf.clip_by_value(depth, 0., 10.)

        return super().update_state(y_true, y_pred, sample_weight)