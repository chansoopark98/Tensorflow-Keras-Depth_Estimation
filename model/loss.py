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

    """Test 1 loss """
    # def ssim(self, x, y):
    #     C1 = 0.01 ** 2
    #     C2 = 0.03 ** 2

    #     x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    #     y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    #     mu_x = tf.nn.avg_pool2d(x, 3, 1, 'VALID')
    #     mu_y = tf.nn.avg_pool2d(y, 3, 1, 'VALID')

    #     sigma_x = tf.nn.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    #     sigma_y = tf.nn.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    #     sigma_xy = tf.nn.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

    #     SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    #     SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    #     SSIM = SSIM_n / SSIM_d

    #     return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    # def get_smooth_loss(self, disp, img):
    #     norm_disp = disp / ( tf.reduce_mean(disp, [1, 2], keepdims=True) + 1e-7)

    #     grad_disp_x = tf.abs(norm_disp[:, :-1, :, :] - norm_disp[:, 1:, :, :])
    #     grad_disp_y = tf.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])

    #     grad_img_x = tf.abs(img[:, :-1, :, :] - img[:, 1:, :, :])
    #     grad_img_y = tf.abs(img[:, :, :-1, :] - img[:, :, 1:, :])

    #     weight_x = tf.exp(-tf.reduce_mean(grad_img_x, 3, keepdims=True))
    #     weight_y = tf.exp(-tf.reduce_mean(grad_img_y, 3, keepdims=True))

    #     smoothness_x = grad_disp_x * weight_x
    #     smoothness_y = grad_disp_y * weight_y

    #     # smooth_loss = tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)
    #     x_sum = tf.reduce_sum(smoothness_x) / tf.cast(tf.reduce_prod(tf.shape(smoothness_x)[1:]), tf.float32)
    #     y_sum = tf.reduce_sum(smoothness_y) / tf.cast(tf.reduce_prod(tf.shape(smoothness_y)[1:]), tf.float32)
    #     smooth_loss = x_sum + y_sum
    #     return smooth_loss
    
    # def depth_loss(self, y_true, y_pred):
    #     y_true = tf.cast(y_true, tf.float32)
    #     y_pred = tf.cast(y_pred, tf.float32)
        
    #     l1_loss = tf.reduce_mean(tf.abs(y_pred - y_true), axis=-1) #, keepdims=True
    #     ssim_loss = tf.reduce_mean(self.ssim(y_pred, y_true), axis=-1) # , keepdims=True


    #     projection_loss = (0.85 * ssim_loss) + (0.15 * l1_loss)
    #     projection_loss = tf.reduce_sum(projection_loss) / tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)

    #     smooth_loss = self.get_smooth_loss(y_pred, y_true)
    #     loss = projection_loss + smooth_loss
    #     return loss

    # """Test 2 loss """
    # def depth_loss(self, y_true, y_pred):
    #     y_true = tf.cast(y_true, tf.float32)
    #     y_pred = tf.cast(y_pred, tf.float32)
    #     # # Point-wise depth
    #     l_depth = tf.reduce_mean(tf.abs(y_true - y_pred))

    #     # # Edges
    #     # dy_true, dx_true = tf.image.image_gradients(y_true)
    #     # dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    #     # l_edges = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    #     # Structural similarity (SSIM) index
    #     # l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, self.max_depth_val)) * 0.5, 0, 1)
    #     l_ssim = tf.reduce_mean(K.clip((1 - tf.image.ssim(y_true, y_pred, self.max_depth_val, filter_size=3, k1=0.01**2, k2=0.03**2)), 0., 1.))

        

    #     # # Weights
    #     # w1 = 1.0
    #     # w2 = 1.0
    #     # w3 = 0.1
    #     # (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth)
    #     return l_ssim + (l_depth * tf.cast(0.1, tf.float32))

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
        l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

        # Weights
        w1 = 1.0
        w2 = 1.0
        w3 = theta

        return (w1 * l_ssim) + (w2 * K.mean(l_edges)) + (w3 * K.mean(l_depth))