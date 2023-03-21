import os
import glob
import cv2
import argparse
import tensorflow as tf
from model.model_builder import ModelBuilder
from utils.load_datasets import GenerateDatasets
import matplotlib.pyplot as plt
K = tf.keras.backend

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",          type=int,    help="Evaluation batch size",
                    default=1)
parser.add_argument("--image_format",           type=str,    help="Image data format (e.g. jpg)",
                    default='png')
parser.add_argument("--image_size",          type=tuple,  help="Model image size (input resolution)",
                    default=(480, 640))
parser.add_argument("--threshold",           type=float,  help="Post processing confidence threshold",
                    default=0.5)
parser.add_argument("--checkpoint_dir",      type=str,    help="Setting the model storage directory",
                    default='./checkpoints/')
parser.add_argument("--weight_name",         type=str,    help="Saved model weights directory",
                    default='0320/_Bs-8_Ep-30_Lr-0.0001_ImSize-480_Opt-adam_multi-gpu_0320_test_efficientv2s_cbam_best_ssim.h5')
# 0317/_Bs-8_Ep-50_Lr-0.0002_ImSize-480_Opt-adam_multi-gpu_0317_EfficientV2S_TEST_best_val_loss.h5
args = parser.parse_args()

if __name__ == '__main__':
    data_loader = GenerateDatasets(data_dir='./datasets/', batch_size=args.batch_size, image_size=args.image_size, dataset_name='nyu_depth_v2') 

    test_data = data_loader.get_testData(test_data=data_loader.train_data)

    # Set target transforms
    model = ModelBuilder(image_size=args.image_size).build_model()
    model.load_weights(args.checkpoint_dir + args.weight_name)
    model.summary()

    # # Set colormap
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
    # plt.colorbar(sm)

    for img, depth in test_data.take(100):
        pred = model.predict(img)
        # pred *= 10.
        # depth *= 10.
        # depth *= 100.
        # depth = 1000. / depth
        # pred = 1000. / pred
        # depth = 1000. / depth

        # Structural similarity (SSIM) index
        # l_ssim = K.mean(K.clip((1 - tf.image.ssim(y_true, y_pred, 1.0)) * 0.5, 0, 1))
        ssim_value = tf.image.ssim(depth, pred, 3., filter_size=7, k1=0.01 ** 2, k2=0.03 ** 2)
        print(tf.reduce_mean(ssim_value))

        divide_shape = tf.cast(tf.reduce_prod(tf.shape(depth)[1:]), tf.float32)
        print('divide', divide_shape)

        rows = 1
        cols = 4
        fig = plt.figure()
        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(img[0], cmap='plasma', vmin=0.0, vmax=2.0)
        ax0.set_title('img')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(pred[0], cmap='plasma', vmin=0.0, vmax=2.0)
        ax0.set_title('pred_depth')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 3)
        ax0.imshow(depth[0], cmap='plasma', vmin=0.0, vmax=2.0)
        ax0.set_title('gt')
        ax0.axis("off")

        fig.subplots_adjust(right=1.0)
        cbar_ax = fig.add_axes([0.82, 0.35, 0.02, 0.3])
        fig.colorbar(sm, cax=cbar_ax)
    
        plt.show()