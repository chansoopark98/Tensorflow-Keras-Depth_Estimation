import tensorflow as tf
K = tf.keras.backend

class DepthEstimationLoss():
    def __init__(self, global_batch_size, distribute_mode=False):
        self.global_batch_size = global_batch_size
        self.distribute_mode = distribute_mode
        self.max_depth_val = 1.0
        self.ssim_loss_weight = 0.85
        self.l1_loss_weight = 0.1
        self.rmse_loss_weight = 0.1
        self.edge_loss_weight = 0.9

    def si_loss(self, y_true, y_pred):
        g = tf.abs(tf.math.log(y_true) - tf.math.log(y_pred))        
        Dg = tf.math.reduce_variance(g, axis=-1) + 0.15 * tf.math.pow(tf.math.reduce_mean(g, axis=-1), 2)
        return 10 * tf.math.sqrt(Dg)
    
    def depth_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Edges
        # Point-wise depth
        l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

        # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

        # Structural similarity (SSIM) index
        l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, 1.0)), 0, 1)

        # Weights
        w1 = 1.0
        w2 = 1.0
        w3 = 0.1

        return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))