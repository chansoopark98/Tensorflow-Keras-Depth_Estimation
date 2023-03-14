from azure_kinect import PyAzureKinectCamera
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from datetime import datetime
import csv

def depth_inpaint(depth, max_value=10, missing_value=0) -> np.ndarray:
    depth = np.where(depth > max_value, 0, depth)

    depth = cv2.copyMakeBorder(depth, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (depth == missing_value).astype(np.uint8)

    scale = np.abs(depth).max()
    depth = depth.astype(np.float32) / scale
    depth = cv2.inpaint(depth, mask, 5, cv2.INPAINT_NS)

    depth = depth[1:-1, 1:-1]
    depth = depth * scale

    return depth

now = datetime.now()
current_time = now.strftime('%Y_%m_%d_%H_%M_%S')

dir_name = 'kinect_capture_data'
dir_path = './{0}/{1}/'.format(dir_name, current_time)

rgb_path = dir_path + 'rgb/'
depth_path = dir_path + 'depth/'

os.makedirs(rgb_path, exist_ok=True)
os.makedirs(depth_path, exist_ok=True)

rgb_resolution = '1536'
rgb_resolution
camera = PyAzureKinectCamera(resolution=rgb_resolution)
camera.capture()
camera_intrinsic = list(camera.get_color_intrinsic_matrix())

with open(os.path.join(dir_path, 'camera_information.csv'),'w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow([int(rgb_resolution)])
    writer.writerow(camera_intrinsic)

idx = 0
if __name__ == "__main__":
    while True:
        camera.capture()
        rgb = camera.get_color()
        rgb = rgb[:, :, :3].astype(np.uint8)
        depth = camera.get_transformed_depth()

        rgb = cv2.resize(rgb, (640, 480), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)

        depth = depth / 1000.
        depth = depth_inpaint(depth=depth)
        
        vis_depth = depth.copy() * 100
        
        # vis_depth = depth_inpaint(depth=vis_depth, max_value=100)
        
        vis_depth = np.expand_dims(vis_depth.astype(np.uint8), axis=-1)
        vis_depth = np.concatenate([vis_depth, vis_depth, vis_depth], axis=-1)
        
        concat = cv2.hconcat([rgb, vis_depth])
        cv2.imshow('test', concat)
        key = cv2.waitKey(300)
        
        
        if key == ord('q'):
            print(key)
            break
        elif key == ord('d'):
            idx += 1
            print(key)
            print('current index is : {0}'.format(idx))

            cv2.imwrite(rgb_path + '_' + dir_name + '_' + current_time + '_rgb_{0}.jpg'.format(idx), rgb )
            np.save(depth_path + '_' + dir_name + '_' + current_time + '_depth_{0}.npy'.format(idx), depth)