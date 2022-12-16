import os
import glob
import cv2
import argparse
import tensorflow as tf
from model.model_builder import ModelBuilder
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",          type=int,    help="Evaluation batch size",
                    default=1)
parser.add_argument("--image_dir",           type=str,    help="Image directory",
                    default='./inputs/')
parser.add_argument("--image_format",           type=str,    help="Image data format (e.g. jpg)",
                    default='png')
parser.add_argument("--image_size",          type=tuple,  help="Model image size (input resolution)",
                    default=(256, 256))
parser.add_argument("--threshold",           type=float,  help="Post processing confidence threshold",
                    default=0.5)
parser.add_argument("--checkpoint_dir",      type=str,    help="Setting the model storage directory",
                    default='./checkpoints/')
parser.add_argument("--weight_name",         type=str,    help="Saved model weights directory",
                    default='1215/_Bs-16_Ep-100_Lr-0.001_ImSize-256_Opt-adamW_multi-gpu_1215_test_loss-single-adamW_best_loss.h5')

args = parser.parse_args()


if __name__ == '__main__':
    image_list = os.path.join(args.image_dir, '*.' + args.image_format)
    image_list = glob.glob(image_list)

    result_dir = args.image_dir + '/results/'
    os.makedirs(result_dir, exist_ok=True)

    # Set target transforms
    model = ModelBuilder(image_size=args.image_size).build_model()
    model.load_weights(args.checkpoint_dir + args.weight_name)
    model.summary()

    for i in range(len(image_list)):
        frame = cv2.imread(image_list[i])

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = tf.image.resize(img, size=args.image_size,
                method=tf.image.ResizeMethod.BILINEAR)

        img = tf.cast(img, tf.float32)
        img /= 255.
        
        img = tf.expand_dims(img, axis=0)

        pred = model.predict(img)


        rows = 1
        cols = 3
        fig = plt.figure()
        ax0 = fig.add_subplot(rows, cols, 1)
        ax0.imshow(img[0])
        ax0.set_title('img')
        ax0.axis("off")

        ax0 = fig.add_subplot(rows, cols, 2)
        ax0.imshow(pred[0])
        ax0.set_title('pred_depth')
        ax0.axis("off")
    
        plt.show()
