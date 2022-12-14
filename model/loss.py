import tensorflow as tf
K = tf.keras.backend

class DepthEstimationLoss():
    def __init__(self, global_batch_size, distribute_mode=False):
        self.global_batch_size = global_batch_size
        self.distribute_mode = distribute_mode
        self.max_depth_val = 1000.0/10.0
        self.ssim_loss_weight = 0.85
        self.l1_loss_weight = 0.1
        self.edge_loss_weight = 0.9
    
    def depth_loss(self, y_true, y_pred):
        # RMSE loss
        rmse_loss = tf.reduce_mean(tf.sqrt(tf.losses.mean_squared_error(y_true, y_pred)))
        
        # Point-wise depth
        # l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)
        l_depth = tf.reduce_mean(tf.math.abs(y_pred - y_true))
        
        
        # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        # l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)
        l_edges = tf.reduce_mean(tf.math.abs(dy_pred - dy_true) + tf.math.abs(dx_pred - dx_true))

        # Structural similarity (SSIM) index
        # l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, self.max_depth_val)) * 0.5, 0, 1)
        l_ssim = tf.clip_by_value((1 - tf.image.ssim(y_true, y_pred, self.max_depth_val)) * 0.5, 0, 1)
        
        # Weights
        w1 = self.ssim_loss_weight
        w2 = self.edge_loss_weight
        w3 = self.l1_loss_weight

        # total_loss = (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth)) + rmse_loss
        total_loss = (w1 * l_ssim) + (w2 * l_edges) + (w3 * l_depth) + rmse_loss
        
        return total_loss