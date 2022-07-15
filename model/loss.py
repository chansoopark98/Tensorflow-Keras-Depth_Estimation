import tensorflow as tf
import tensorflow.keras.backend as K

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
        tf.mathabs(smoothness_y)
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
        self.max_depth_scale = 1.0
        self.ssim_loss_weight = 0.85
        self.l1_loss_weight = 0.1
        self.edge_loss_weight = 0.9
    
    def depth_loss(self, y_true, y_pred):
        l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

        # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

        # Structural similarity (SSIM) index
        l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, self.max_depth_scale)) * 0.5, 0, 1)

        # Weights
        w1 = 1.0
        w2 = 1.0
        w3 = 0.1

        return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))

    def distribute_depth_loss(self, y_true, y_pred):
        l_depth = K.abs(y_pred - y_true)
        l_depth = tf.reduce_sum(l_depth) * (1. / self.global_batch_size)
        l_depth /= tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)

        # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)

        l_edges = K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true)
        l_edges = tf.reduce_sum(l_edges) * (1. / self.global_batch_size)
        l_edges /= tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)

        # Structural similarity (SSIM) index
        l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, self.max_depth_scale)) * 0.5, 0, 1)
        l_ssim = tf.reduce_sum(l_ssim) * (1. / self.global_batch_size)
        l_ssim /= tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)

        # Weights
        w1 = 0.85
        w2 = 0.9
        w3 = 0.1

        return (w1 * l_ssim) + (w2 * l_edges) + (w3 * l_depth)
            

        # if self.distribute_mode:
        #     loss = tf.reduce_sum(loss) * (1. / self.global_batch_size)
        #     loss /= tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)

        
