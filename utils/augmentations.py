import tensorflow as tf
import tensorflow_addons as tfa
from typing import Union

class Augmentation(object):
    def __init__(self, image_size: tuple, max_crop_scale: float):
        self.image_size = image_size
        self.max_crop_scale = max_crop_scale
        self.min_depth = 0.01
        self.max_depth = 1.0


    def depth_norm(self, depth: tf.Tensor, max_depth: float) -> tf.Tensor:
        return max_depth / depth

    def normalize(self, image: tf.Tensor, depth: tf.Tensor) -> Union[tf.Tensor, tf.Tensor]:
        image /= 255.
        depth /= 10.
        
        depth_zero_mask = tf.cast(tf.math.equal(depth, 0.), tf.float32)
        depth = tf.clip_by_value(depth, 0.05, 1.0)

        inverse_depth_data = 1 / depth

        # Reset the masked areas (previously zero) to zero
        inverse_depth_data *= (1.0 - depth_zero_mask)

        # Clip the remaining values
        inverse_depth_data = tf.clip_by_value(inverse_depth_data, 0., 20.)

        return image, inverse_depth_data

        
    def random_gamma(self, image: tf.Tensor, depth: tf.Tensor) -> Union[tf.Tensor, tf.Tensor]:
        random_gamma = tf.random.uniform([], 0.8, 1.2)
        image = image ** random_gamma
        image = tf.clip_by_value(image, 0, 255)
        return (image, depth)

    def random_brightness(self, image: tf.Tensor, depth: tf.Tensor) -> Union[tf.Tensor, tf.Tensor]:
        random_bright = tf.random.uniform([], 0.7, 1.3)
        image = image * random_bright
        image = tf.clip_by_value(image, 0, 255)
        return (image, depth)

    def random_color(self, image: tf.Tensor, depth: tf.Tensor) -> Union[tf.Tensor, tf.Tensor]:
        colors = tf.random.uniform([3], 0.8, 1.2)
        white = tf.ones([self.image_size[0], self.image_size[1]])
        color_image = tf.stack([white * colors[i] for i in range(3)], axis=2)
        image *= color_image
        image = tf.clip_by_value(image, 0, 255)
        return (image, depth)


    def random_crop(self, image: tf.Tensor, depth: tf.Tensor) -> Union[tf.Tensor, tf.Tensor]:
        scale = tf.random.uniform([], 1.0, self.max_crop_scale)

        new_h = self.image_size[0] * scale
        new_w = self.image_size[1] * scale

        image = tf.image.resize(image, size=(new_h, new_w),
                            method=tf.image.ResizeMethod.BILINEAR)
        depth = tf.image.resize(depth, size=(new_h, new_w),
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        rgbd = tf.concat([image, depth], axis=-1)
        rgbd = tf.image.random_crop(
            rgbd, (self.image_size[0], self.image_size[1], 4))

        image = rgbd[:, :, :3]
        depth = rgbd[:, :, 3:]

        return (image, depth)
    
    def random_rotate(self, image: tf.Tensor, depth: tf.Tensor) -> Union[tf.Tensor, tf.Tensor]:
        # Degrees to Radian
        upper = 30 * (3.14 / 180.0)

        rand_degree = tf.random.uniform([], minval=-upper, maxval=upper)

        image = tfa.image.rotate(image, rand_degree, interpolation='bilinear')
        depth = tfa.image.rotate(depth, rand_degree)

        return (image, depth)

    
    def horizontal_flip(self, image: tf.Tensor, depth: tf.Tensor) -> Union[tf.Tensor, tf.Tensor]:
        image = tf.image.flip_left_right(image=image)
        depth = tf.image.flip_left_right(image=depth)

        return (image, depth)