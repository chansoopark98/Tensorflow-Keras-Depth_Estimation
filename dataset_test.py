import tensorflow as tf
from utils.load_datasets import GenerateDatasets
# from utils.plot_generator import plot_generator
import numpy as np
import matplotlib.pyplot as plt

dataset = GenerateDatasets(data_dir='./datasets/', image_size=(480, 640), batch_size=1, dataset_name='nyu_depth_v2')

test_data = dataset.get_trainData(dataset.train_data)

tf.config.set_soft_device_placement(True)

max_scale = 10
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
plt.colorbar(sm)

if __name__ == "__main__":
    max_depth_scale = 0
    min_depth_scale = 0
    i = 1
    for img, depth in test_data.take(dataset.number_train):
        # depth = tf.cast(depth, tf.float32)
        
        rows = 1
        cols = 2
        img = img[0]
        fig = plt.figure()
        
        depth = tf.where(tf.math.is_inf(depth), 0., depth)
        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(img, vmin=0.0, vmax=max_scale)
        ax0.set_title('Image')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(depth[0], vmin=0.0, vmax=max_scale)
        ax0.set_title('Depth map')
        ax0.axis("off")

        plt.show()
        if i % 100 == 0:
            print(i)
        i+=1
        
        is_inf = tf.reduce_any(tf.math.is_inf(depth))
        
        is_nan = tf.reduce_any(tf.math.is_nan(depth))
        
        if is_inf is True:
            print('inf')
        if is_nan is True:
            print('nan')
        max_depth = tf.reduce_max(depth)
        if max_depth >= max_depth_scale:
            max_depth_scale = max_depth
        min_depth = tf.reduce_min(depth)
        if min_depth <= min_depth_scale:
            min_depth_scale = min_depth

    print('max_depth_scale', max_depth_scale)
    print('min_depth_scale', min_depth_scale)