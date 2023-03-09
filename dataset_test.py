import tensorflow as tf
from utils.load_datasets import GenerateDatasets
# from utils.plot_generator import plot_generator
import numpy as np
import matplotlib.pyplot as plt

dataset = GenerateDatasets(data_dir='./datasets/', image_size=(720, 1280), batch_size=1, dataset_name='nyu_depth_v2')

test_data = dataset.get_trainData(dataset.train_data)

tf.config.set_soft_device_placement(True)

max_scale = 1.
sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
plt.colorbar(sm)
if __name__ == "__main__":

    i = 1
    for img, depth in test_data.take(100):

        depth = tf.cast(depth, tf.float32)
               
        rows = 1
        cols = 2
        img = img[0]
        fig = plt.figure()
        
        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(img, vmin=0.0, vmax=max_scale)
        ax0.set_title('Image')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(depth[0], vmin=0.0, vmax=max_scale)
        ax0.set_title('Depth map')
        ax0.axis("off")

        # fig.subplots_adjust(right=1.0)
        # cbar_ax = fig.add_axes([0.72, 0.35, 0.02, 0.3])
        # fig.colorbar(sm, cax=cbar_ax)

        plt.show()
        
        # plt.savefig('./plt_outputs/output_{0}'.format(i), dpi=300)
        # i += 1