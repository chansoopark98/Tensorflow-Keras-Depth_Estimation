import tensorflow as tf
from utils.dataset_generator import DatasetGenerator
from utils.plot_generator import plot_generator
import matplotlib.pyplot as plt

dataset = DatasetGenerator(data_dir='./datasets/', image_size=(256, 256), batch_size=1)

train_data = dataset.get_trainData()
valid_data = dataset.get_validData()

if __name__ == "__main__":
    for img, depth in train_data.take(100):
        depth = tf.expand_dims(depth, axis=-1)
        # img = tf.image.resize(img, (self.image_size[0], self.image_size[1]), tf.image.ResizeMethod.BILINEAR)
        # depth = tf.image.resize(depth, (self.image_size[0], self.image_size[1]), tf.image.ResizeMethod.BILINEAR)

        # Format
        img = tf.image.convert_image_dtype(img / 255., dtype=tf.float32)
        depth = tf.image.convert_image_dtype(depth/255., dtype=tf.float32)
        # depth = tf.cast(depth / 255.0, tf.float32)
        # Normalize the depth values (in cm)
        # norm_depth = 1000 / tf.clip_by_value(norm_depth * 1000, 10, 1000)
        # norm_depth = 1. / depth
        norm_depth = tf.math.divide_no_nan(1000., depth*1000)
        norm_depth /= 1000.


        rows = 1
        cols = 2
        fig = plt.figure()
        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(img)
        ax0.set_title('img')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(norm_depth)
        ax0.set_title('depth')
        ax0.axis("off")

    
        plt.show()



