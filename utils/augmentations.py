import tensorflow as tf
import tensorflow_addons as tfa
from typing import Union

class Augmentation(object):
    def __init__(self, image_size: tuple, max_crop_scale: float):
        self.image_size = image_size
        self.max_crop_scale = max_crop_scale
        self.min_depth = 0.05
        self.max_depth = 1.0

    def add_gaussian_noise(self, image, depth, mean=0.0, stddev=10.):
        noise = tf.random.normal(shape=tf.shape(image), mean=mean, stddev=stddev)
        noisy_image = image + noise
        noisy_image = tf.clip_by_value(noisy_image, 0.0, 255.)
        return noisy_image, depth

    def random_scaling(self, image, depth, scale_range=(0.8, 1.2)):
        scale = tf.random.uniform([], minval=scale_range[0], maxval=scale_range[1])
        original_height, original_width, _ = image.shape
        new_height = tf.cast(tf.cast(original_height, tf.float32) * scale, tf.int32)
        new_width = tf.cast(tf.cast(original_width, tf.float32) * scale, tf.int32)

        # Resize the image and depth map using the random scale
        scaled_image = tf.image.resize(image, (new_height, new_width))
        scaled_depth_map = tf.image.resize(depth, (new_height, new_width))

        if scale < 1.0:
            # Pad the scaled image and depth map to match the target size
            padding_height = self.image_size[0] - new_height
            padding_width = self.image_size[1] - new_width
            padding_top = padding_height // 2
            padding_bottom = padding_height - padding_top
            padding_left = padding_width // 2
            padding_right = padding_width - padding_left

            resized_image = tf.pad(scaled_image, [[padding_top, padding_bottom], [padding_left, padding_right], [0, 0]])
            resized_depth_map = tf.pad(scaled_depth_map, [[padding_top, padding_bottom], [padding_left, padding_right], [0, 0]])
        else:
            concat_img = tf.concat([scaled_image, scaled_depth_map], axis=-1)
            concat_img = tf.image.random_crop(
                    concat_img, (self.image_size[0], self.image_size[1], 4))

            resized_image = concat_img[:, :, :3]
            resized_depth_map = concat_img[:, :, 3:]
            
        return resized_image, resized_depth_map

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
        upper = 10 * (3.14 / 180.0)

        rand_degree = tf.random.uniform([], minval=-upper, maxval=upper)

        image = tfa.image.rotate(image, rand_degree, interpolation='bilinear')
        depth = tfa.image.rotate(depth, rand_degree)

        return (image, depth)

    
    def horizontal_flip(self, image: tf.Tensor, depth: tf.Tensor) -> Union[tf.Tensor, tf.Tensor]:
        image = tf.image.flip_left_right(image=image)
        depth = tf.image.flip_left_right(image=depth)

        return (image, depth)