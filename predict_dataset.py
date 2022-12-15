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
                    default=1)
parser.add_argument("--image_format",           type=str,    help="Image data format (e.g. jpg)",
                    default='png')
parser.add_argument("--image_size",          type=tuple,  help="Model image size (input resolution)",
                    default=(256, 256))
parser.add_argument("--threshold",           type=float,  help="Post processing confidence threshold",
                    default=0.5)
parser.add_argument("--checkpoint_dir",      type=str,    help="Setting the model storage directory",
                    default='./checkpoints/')
parser.add_argument("--weight_name",         type=str,    help="Saved model weights directory",
                    default='1215/_Bs-32_Ep-50_Lr-0.001_ImSize-256_Opt-adamW_multi-gpu_1215_test-model_best_loss.h5')

args = parser.parse_args()


if __name__ == '__main__':
    data_loader = GenerateDatasets(data_dir='./datasets/', batch_size=1, image_size=args.image_size, dataset_name='nyu_depth_v2') 

    test_data = data_loader.get_testData(test_data=data_loader.test_data)

    

    # Set target transforms
    model = ModelBuilder(image_size=args.image_size).build_model()
    model.load_weights(args.checkpoint_dir + args.weight_name)
    model.summary()

    for img, depth in test_data.take(100):
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

        ax0 = fig.add_subplot(rows, cols, 3)
        ax0.imshow(depth[0])
        ax0.set_title('gt')
        ax0.axis("off")
    
        plt.show()
