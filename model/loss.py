import tensorflow as tf
K = tf.keras.backend

class DepthEstimationLoss():
    def __init__(self, global_batch_size, distribute_mode=False):
        self.global_batch_size = global_batch_size
        self.distribute_mode = distribute_mode
        self.max_depth_val = 1.0
        self.ssim_loss_weight = 1.0
        self.l1_loss_weight = 0.1
        self.edge_loss_weight = 1.0



    def ssim_loss(self, y_true, y_pred):
        # Convert the images to tensors
        y_true = K.batch_flatten(K.permute_dimensions(y_true, (2, 0, 1)))
        y_pred = K.batch_flatten(K.permute_dimensions(y_pred, (2, 0, 1)))
        
        # Compute SSIM
        mu_x = K.mean(y_pred)
        mu_y = K.mean(y_true)
        sigma_x = K.std(y_pred)
        sigma_y = K.std(y_true)
        sigma_xy = K.mean((y_pred - mu_x) * (y_true - mu_y))
        c1 = K.constant(0.01 ** 2, dtype=tf.float32)
        c2 = K.constant(0.03 ** 2, dtype=tf.float32)
        ssim = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2) / (mu_x ** 2 + mu_y ** 2 + c1) / (sigma_x ** 2 + sigma_y ** 2 + c2)
        
        # Return negative SSIM as the loss
        return -tf.reduce_mean(ssim)

    def depth_loss(self, y_true, y_pred, theta=0.1, maxDepthVal=10.0):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        # Point-wise depth
        l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

        # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

        # Structural similarity (SSIM) index
        # l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)
        ssim = tf.image.ssim(y_true , y_pred , maxDepthVal)
        # ssim = self.ssim_loss(y_true, y_pred)
        l_ssim = K.clip((1. - ssim) * 0.5, 0, 1)

        # Weights
        w1 = 1.0
        w2 = 0.4 # 0.5
        w3 = 0.2 # 0.3

        return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))
        # return (w1 * l_ssim) + (w3 * K.mean(l_depth))

    # def custom_loss(self, y_true, y_pred):
    #     y_true = tf.cast(y_true, tf.float32)
    #     y_pred = tf.cast(y_pred, tf.float32)
    #     # Mean absolute error loss
    #     mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    #     # Structural similarity index loss
    #     ssim_loss = 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)
    #     ssim_loss = tf.reduce_mean(ssim_loss)

    #     # Log-cosh loss
    #     logcosh_loss = tf.reduce_mean(tf.math.log(tf.cosh(y_pred - y_true)))
    #     # logcosh_loss = tf.keras.losses.log_cosh(y_true=y_true, y_pred=y_pred)

    #     # BerHu loss
    #     huber_delta = 0.5
    #     huber_loss = tf.where(tf.abs(y_true - y_pred) < huber_delta, 0.5 * tf.square(y_true - y_pred), huber_delta * tf.abs(y_true - y_pred) - 0.5 * huber_delta**2)
    #     huber_loss = tf.reduce_mean(huber_loss)

    #     # Combine the losses with weighting factors
    #     total_loss = (0.1 * mae_loss) + (1.0 * ssim_loss) + (0.2 * logcosh_loss) + (0.1 * huber_loss)

    #     return total_loss
    
    def custom_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Structural similarity index loss
        ssim_loss = 1 - tf.image.ssim(y_true, y_pred, max_val=1.0)
        ssim_loss = tf.reduce_mean(ssim_loss)

        # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        edge_loss = tf.reduce_mean(tf.abs(dy_pred - dy_true) + tf.abs(dx_pred - dx_true))
        
        # Log-cosh loss
        logcosh_loss = tf.reduce_mean(tf.math.log(tf.cosh(y_pred - y_true)))
        # logcosh_loss = tf.keras.losses.log_cosh(y_true=y_true, y_pred=y_pred)

        # BerHu loss
        huber_delta = 0.5
        huber_loss = tf.where(tf.abs(y_true - y_pred) < huber_delta, 0.5 * tf.square(y_true - y_pred), huber_delta * tf.abs(y_true - y_pred) - 0.5 * huber_delta**2)
        huber_loss = tf.reduce_mean(huber_loss)

        # Combine the losses with weighting factors
        total_loss = ssim_loss + (0.2 * edge_loss) + logcosh_loss + huber_loss

        return total_loss