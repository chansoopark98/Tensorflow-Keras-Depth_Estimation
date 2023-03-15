import os
import glob
import cv2
import argparse
import tensorflow as tf
from model.model_builder import ModelBuilder
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",          type=int,    help="Evaluation batch size",
                    default=1)
parser.add_argument("--image_dir",           type=str,    help="Image directory",
                    default='./inputs/')
parser.add_argument("--image_format",           type=str,    help="Image data format (e.g. jpg)",
                    default='png')
parser.add_argument("--image_size",          type=tuple,  help="Model image size (input resolution)",
                    default=(480, 640))
parser.add_argument("--threshold",           type=float,  help="Post processing confidence threshold",
                    default=0.5)
parser.add_argument("--checkpoint_dir",      type=str,    help="Setting the model storage directory",
                    default='./checkpoints/')
parser.add_argument("--weight_name",         type=str,    help="Saved model weights directory",
                    default='0315/_Bs-32_Ep-50_Lr-0.0002_ImSize-480_Opt-adam_multi-gpu_0315_230314_EfficientV2B0_customLoss_480x640_adam_lossFactor_best_rmse.h5')

args = parser.parse_args()


def colorize(image: np.ndarray, clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


if __name__ == '__main__':
    image_list = os.path.join(args.image_dir, '*.' + args.image_format)
    image_list = glob.glob(image_list)

    result_dir = args.image_dir + '/results/'
    os.makedirs(result_dir, exist_ok=True)

    # Set target transforms
    model = ModelBuilder(image_size=args.image_size).build_model()
    model.load_weights(args.checkpoint_dir + args.weight_name)
    model.summary()

    cap = cv2.VideoCapture(0)

    # 프레임을 정수형으로 형 변환
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))	# 영상의 넓이(가로) 프레임
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))	# 영상의 높이(세로) 프레임
    
    frame_size = (frameWidth, frameHeight)
    print('frame_size={}'.format(frame_size))

    frameRate = 33

    
    while True:
        # 한 장의 이미지(frame)를 가져오기
        # 영상 : 이미지(프레임)의 연속
        # 정상적으로 읽어왔는지 -> retval
        # 읽어온 프레임 -> frame
        retval, frame = cap.read()
        if not(retval):	# 프레임정보를 정상적으로 읽지 못하면
            break  # while문을 빠져나가기
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = tf.image.resize(img, size=args.image_size,
                method=tf.image.ResizeMethod.BILINEAR)

        img = tf.cast(img, tf.float32)
        img /= 255.
        
        img = tf.expand_dims(img, axis=0)

        pred = model.predict(img)
        pred = pred[0]

        pred = pred * 1000
        pred = pred.astype(np.uint8)
        # pred = cv2.applyColorMap(pred, cv2.COLORMAP_PLASMA)
        pred = colorize(pred, (0, 1000), colormap=cv2.COLORMAP_JET)

        cv2.imshow('test', pred)
        cv2.waitKey(1)