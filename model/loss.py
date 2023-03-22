import tensorflow as tf
K = tf.keras.backend

class DepthEstimationLoss():
    def __init__(self, global_batch_size, distribute_mode=False):
        self.global_batch_size = global_batch_size
        self.distribute_mode = distribute_mode

    def gradient_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        # Compute Sobel operators (3x3 kernel) for both x and y directions
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=tf.float32)
        sobel_y = tf.transpose(sobel_x)
        
        sobel_x = tf.reshape(sobel_x, (3, 3, 1, 1))
        sobel_y = tf.reshape(sobel_y, (3, 3, 1, 1))
        
        # Compute gradient maps for both ground truth and predicted depth maps
        y_true_dx = tf.nn.conv2d(y_true, sobel_x, strides=1, padding='SAME')
        y_true_dy = tf.nn.conv2d(y_true, sobel_y, strides=1, padding='SAME')
        
        y_pred_dx = tf.nn.conv2d(y_pred, sobel_x, strides=1, padding='SAME')
        y_pred_dy = tf.nn.conv2d(y_pred, sobel_y, strides=1, padding='SAME')
        
        # Compute L1 loss between gradient maps
        loss_dx = tf.abs(y_true_dx - y_pred_dx)
        loss_dy = tf.abs(y_true_dy - y_pred_dy)
        
        return tf.reduce_mean(loss_dx + loss_dy)

    def edge_smooth_loss(self, y_true, y_pred):
        # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
        weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

        # Depth smoothness
        smoothness_x = dx_pred * weights_x
        smoothness_y = dy_pred * weights_y

        depth_smoothness_loss = tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(
            tf.abs(smoothness_y)
        )
        return depth_smoothness_loss
    
    def ssim_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor, max_val: float = 1.0 / 0.05, filter_sigma: float = 1.5, k1: float = 0.01, k2: float = 0.03) -> tf.Tensor:
        ssim = tf.image.ssim(y_true, y_pred, max_val, filter_size=11,
                            filter_sigma=filter_sigma, k1=k1, k2=k2)
        ssim_loss = 1.0 - tf.reduce_mean(ssim)
        return ssim_loss
    
    def depth_to_normals(self, depth: tf.Tensor) -> tf.Tensor:
        depth_dy, depth_dx = tf.image.image_gradients(depth)
        normals = tf.stack([-depth_dx, -depth_dy, tf.ones_like(depth)], axis=-1)
        normals = tf.nn.l2_normalize(normals, axis=-1)
        return normals
    
    def normal_map_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        true_normals = self.depth_to_normals(y_true)
        pred_normals = self.depth_to_normals(y_pred)
        
        cos_similarity = tf.reduce_sum(true_normals * pred_normals, axis=-1)
        loss = 1 - tf.reduce_mean(cos_similarity)
        return loss

    def custom_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))

        # Structural similarity index loss
        ssim_loss = self.ssim_loss(y_true=y_true, y_pred=y_pred)

        # Edge smooth
        edge_smooth_loss = self.edge_smooth_loss(y_true=y_true, y_pred=y_pred)

        # gradient_loss
        gradient_loss = self.gradient_loss(y_true=y_true, y_pred=y_pred)

        # normal_map_loss
        normal_map_loss = self.normal_map_loss(y_true=y_true, y_pred=y_pred)

        total_loss = (1.0 * ssim_loss) +\
             (1.0 * mae_loss) +\
                  (1.0 * edge_smooth_loss) +\
                  (1.0 * gradient_loss) +\
                  normal_map_loss
        return total_loss