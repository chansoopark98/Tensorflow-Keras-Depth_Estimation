import tensorflow as tf
K = tf.keras.backend

class DepthEstimationLoss():
    def __init__(self, global_batch_size, distribute_mode=False):
        self.global_batch_size = global_batch_size
        self.distribute_mode = distribute_mode
        self.max_depth_val = 1.0
        self.ssim_loss_weight = 1.0
        self.l1_loss_weight = 0.1
        self.rmse_loss_weight = 0.1
        self.edge_loss_weight = 1.0
    
    def depth_loss(self, y_true, y_pred):
        # RMSE loss
        # rmse_loss = K.mean(tf.sqrt(tf.losses.mean_squared_error(y_true, y_pred)))
        
        rmse_loss = K.mean(tf.sqrt(K.abs(tf.math.square(y_pred - y_true))), axis=-1)
        # Point-wise depth
        l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)
        
        # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

        # Structural similarity (SSIM) index
        l_ssim = K.mean(K.clip((1 - tf.image.ssim(y_true, y_pred, self.max_depth_val)) * 0.5, 0, 1))
        
        print('rmse_loss', rmse_loss)
        print('l_depth', l_depth)
        print('l_edges', l_edges)
        print('l_ssim', l_ssim)
        # Weights
        w1 = self.ssim_loss_weight
        w2 = self.edge_loss_weight
        w3 = self.l1_loss_weight
        w4 = self.rmse_loss_weight

        # total_loss = (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth)) + rmse_loss
        total_loss = (w1 * l_ssim) + (w2 * l_edges) + (w3 * l_depth) + (w4 * rmse_loss)
        return total_loss