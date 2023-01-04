import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import glob

def depth_inpaint(depth, max_value=10, missing_value=0):
    depth = np.where(depth > 10, 0, depth)

    depth = cv2.copyMakeBorder(depth, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (depth == missing_value).astype(np.uint8)

    scale = np.abs(depth).max()
    depth = depth.astype(np.float32) / scale
    depth = cv2.inpaint(depth, mask, 1, cv2.INPAINT_NS)

    depth = depth[1:-1, 1:-1]
    depth = depth * scale

    return depth

data_dir = './datasets/diode_raw_datasets/train/'

locations = glob.glob(data_dir + '*')

# 실내/ 실외 구분
for location in locations:
    scenes = glob.glob(location + '/*')
    # Scene 구분
    for scene in scenes:
        scans = glob.glob(scene + '/*')
        # Scane 구분
        for scan in scans:
            data_list = glob.glob(scan + '/*.png')
            # Data 구분
            for data in data_list:
                image = data
                depth = '.' + data.split('.')[1] + '_depth.npy'
                depth = np.load(depth)
                




