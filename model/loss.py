import tensorflow as tf
K = tf.keras.backend

ssim_loss_weight = 0.85
l1_loss_weight = 0.1
edge_loss_weight = 0.9

def depth_loss_function(y_true, y_pred, theta=0.1, maxDepthVal=1.0):
    # Point-wise depth
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    # Edges
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    # Structural similarity (SSIM) index
    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    # Weights
    w1 = 1.0
    w2 = 1.0
    w3 = theta

    return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))



def depth_estimation_loss(y_true, y_pred, max_depth=1.0):
    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
    weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

    # Depth smoothness
    smoothness_x = dx_pred * weights_x
    smoothness_y = dy_pred * weights_y
    
    depth_smoothness_loss = tf.reduce_mean(tf.math.abs(smoothness_x)) + tf.reduce_mean(
        tf.math.abs(smoothness_y)
    )

    # Structural similarity (SSIM) index
    ssim_index = tf.image.ssim(y_true, y_pred, max_val=max_depth, filter_size=7, k1=0.01**2, k2=0.03**2)
    ssim_loss = tf.reduce_mean(1 - ssim_index)

    # Point-wise depth
    l1_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

    loss = (
        (ssim_loss_weight * ssim_loss)
        + (l1_loss_weight * l1_loss)
        + (edge_loss_weight * depth_smoothness_loss)
    )

    return loss

class DepthEstimationLoss():
    def __init__(self, global_batch_size, distribute_mode=False):
        self.global_batch_size = global_batch_size
        self.distribute_mode = distribute_mode
        self.max_depth_val = 1000.0/10.0
        self.theta = 0.1
        self.max_depth_scale = 1.0
        self.ssim_loss_weight = 0.85
        self.l1_loss_weight = 0.1
        self.edge_loss_weight = 0.9
    
    def depth_loss(self, y_true, y_pred):
        # rmse_loss = tf.sqrt(tf.losses.mean_squared_error(y_true, y_pred))

        # Point-wise depth
        l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

        # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

        # Structural similarity (SSIM) index
        l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, self.max_depth_val)) * 0.5, 0, 1)

        # Weights
        w1 = 1.0
        w2 = 1.0
        w3 = self.theta

        total_loss = (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))
        
        # if self.distribute_mode:
        #     total_loss = tf.reduce_sum(total_loss) * (1. / self.global_batch_size)

        return total_loss