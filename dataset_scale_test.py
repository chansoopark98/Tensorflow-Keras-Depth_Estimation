import tensorflow as tf
from utils.load_datasets import GenerateDatasets, DataLoadHandler
import numpy as np
import matplotlib.pyplot as plt



dataset = DataLoadHandler(data_dir='./datasets/', dataset_name='nyu_depth_v2', percentage=100)
test_data = dataset.train_data


sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
# plt.colorbar(sm)

if __name__ == "__main__":
    max_depth_scale = 0
    min_depth_scale = 0
    i = 1
    for sample in test_data.take(dataset.number_train):
        print(i)
        # image = tf.cast(sample['image'], tf.float32)
        depth = tf.cast(sample['depth'], tf.float32)
        depth = tf.expand_dims(depth, axis=-1)

        i += 1
        max_depth = tf.reduce_max(depth)
        if max_depth >= max_depth_scale:
            max_depth_scale = max_depth
        min_depth = tf.reduce_min(depth)
        if min_depth <= min_depth_scale:
            min_depth_scale = min_depth

    print('max_depth_scale', max_depth_scale)
    print('min_depth_scale', min_depth_scale)