import os
import glob
import cv2
import argparse
import tensorflow as tf
from model.model_builder import ModelBuilder
from utils.load_datasets import GenerateDatasets
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",          type=int,    help="Evaluation batch size",
                    default=8)
parser.add_argument("--image_format",           type=str,    help="Image data format (e.g. jpg)",
                    default='png')
parser.add_argument("--image_size",          type=tuple,  help="Model image size (input resolution)",
                    default=(256, 256))
parser.add_argument("--threshold",           type=float,  help="Post processing confidence threshold",
                    default=0.5)
parser.add_argument("--checkpoint_dir",      type=str,    help="Setting the model storage directory",
                    default='./checkpoints/')
parser.add_argument("--weight_name",         type=str,    help="Saved model weights directory",
                    default='1219/_Bs-16_Ep-50_Lr-0.001_ImSize-256_Opt-adamW_multi-gpu_1219_new_model-l1_loss_weight0.5-testNew-sgd_best_loss.h5')

args = parser.parse_args()

if __name__ == '__main__':
    data_loader = GenerateDatasets(data_dir='./datasets/', batch_size=1, image_size=args.image_size, dataset_name='nyu_depth_v2') 

    test_data = data_loader.get_testData(test_data=data_loader.test_data)

    # Set target transforms
    model = ModelBuilder(image_size=args.image_size).build_model()
    model.load_weights(args.checkpoint_dir + args.weight_name)
    model.summary()

    # Set colormap
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=1))
    plt.colorbar(sm)

    for img, depth in test_data.take(100):
        pred = model.predict(img)

        # test loss
        target = tf.cast(depth, tf.float32)
        pred = tf.cast(pred, tf.float32)

        # Edges
        dy_true, dx_true = tf.image.image_gradients(target)
        dy_pred, dx_pred = tf.image.image_gradients(pred)
        weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
        weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

        # Depth smoothness
        smoothness_x = dx_pred * weights_x
        smoothness_y = dy_pred * weights_y

        depth_smoothness_loss = tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(
            tf.abs(smoothness_y)
        )

        # Structural similarity (SSIM) index
        ssim_loss = tf.reduce_mean(
            1
            - tf.image.ssim(
                target, pred, max_val=1.0, filter_size=7, k1=0.01**2, k2=0.03**2
            )
        )
        l1_loss = tf.reduce_mean(tf.square(tf.abs(target - pred)))

        g = tf.abs(tf.math.log(target) - tf.math.log(pred))
        
        
        Dg = tf.math.reduce_variance(g) + 0.15 * tf.math.pow(tf.math.reduce_mean(g), 2)
        
        si_loss = 10 * tf.math.sqrt(Dg)

        # print('l1_loss : {0}'.format(l1_loss))
        # print('ssim_loss : {0}'.format(ssim_loss))
        # print('depth_smoothness_loss : {0}'.format(depth_smoothness_loss))
        print('log target', tf.reduce_mean(tf.math.log(target)))
        print('log pred', tf.reduce_mean(tf.math.log(pred)))
        print(tf.reduce_mean(g))
        print(Dg)
        print('si_loss : {0}'.format(si_loss))


        rows = 1
        cols = 4
        fig = plt.figure()
        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(img[0], cmap='plasma', vmin=0.0, vmax=1.0)
        ax0.set_title('img')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(pred[0], cmap='plasma', vmin=0.0, vmax=1.0)
        ax0.set_title('pred_depth')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 3)
        ax0.imshow(depth[0], cmap='plasma', vmin=0.0, vmax=1.0)
        ax0.set_title('gt')
        ax0.axis("off")

        fig.subplots_adjust(right=1.0)
        cbar_ax = fig.add_axes([0.82, 0.35, 0.02, 0.3])
        fig.colorbar(sm, cax=cbar_ax)
    
        plt.show()
