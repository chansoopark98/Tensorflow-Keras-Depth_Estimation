import matplotlib.pyplot as plt
import cv2
import pyk4a
from pyk4a import Config, PyK4A

import numpy as np

if __name__ == "__main__":

    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_1080P,
            depth_mode=pyk4a.DepthMode.OFF,
            synchronized_images_only=False
        )
    )
    k4a.start()

    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    # Initial setting
    capture = k4a.get_capture().color[:, :, :3]
    crop_x, crop_y, crop_w, crop_h = cv2.selectROI('test', capture)
    rgb_shape = capture.shape

    idx = 0
    while True:
        idx += 1
        if idx == 37:
            break
        capture = k4a.get_capture()

        rgb = capture.color[:, :, :3]
        crop_rgb = rgb[crop_y: crop_y + crop_h, crop_x: crop_x + crop_w]
        
        background = np.ones_like(rgb) * 255
        
        background[crop_y: crop_y + crop_h, crop_x: crop_x + crop_w] = crop_rgb
        
        
        cv2.imshow('test', background)


        file_name = str(idx).zfill(2)
        cv2.imwrite('./image_results/img' + file_name + '.jpg', background)
        if cv2.waitKey(750) == ord('q'): # q를 누르면 종료
            
            break