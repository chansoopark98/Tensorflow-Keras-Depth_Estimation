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
    crop_rgb = capture[crop_y: crop_y + crop_h, crop_x: crop_x + crop_w]
    fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=False, dist2Threshold=2000)

    
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(11, 11))

    dil_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1,3))


    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # 열림 연산 적용 ---②
    
    idx = 0

    # while True:
        # idx += 1
        # if idx == 37:
        #     break

        # capture = k4a.get_capture()
        # rgb = capture.color[:, :, :3]
        # crop_rgb = rgb
        # # crop_rgb = rgb[crop_y: crop_y + crop_h, crop_x: crop_x + crop_w]
        # mask = fgbg.apply(crop_rgb)
        # mask = cv2.dilate(mask,kernel,iterations=3)
        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # # c = max(contours, key = cv2.contourArea)
        # zero = np.zeros(mask.shape)
        # cv2.drawContours(zero, contours, -1, 255, -1) #---set the last parameter to -1

        # zero /= 255
        # zero = np.expand_dims(zero.astype(np.uint8), axis=-1)
        # crop_rgb = crop_rgb.astype(np.uint8)

        # crop_rgb = np.where(zero==0, 255, crop_rgb)

        # # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

        # cv2.imshow('test', mask)

        # if cv2.waitKey(750) == ord('q'): # q를 누르면 종료
        #     break
        
    """
        mask = fgbg.apply(crop_rgb)
        mask = cv2.dilate(mask,kernel,iterations=1)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        # c = max(contours, key = cv2.contourArea)
        zero = np.zeros(mask.shape)
        cv2.drawContours(zero, contours, -1, 255, -1) #---set the last parameter to -1

        zero /= 255
        zero = np.expand_dims(zero.astype(np.uint8), axis=-1)
        crop_rgb = crop_rgb.astype(np.uint8)

        crop_rgb = np.where(zero==0, 255, crop_rgb)
    """

    # --------------------------------------------------------------------
    idx = 0
    while True:
        idx += 1
        if idx == 65:
            break
        capture = k4a.get_capture()

        rgb = capture.color[:, :, :3]
        # crop_rgb = rgb[crop_y: crop_y + crop_h, crop_x: crop_x + crop_w]
        
        # background = np.ones_like(rgb) * 255
        
        # background[crop_y: crop_y + crop_h, crop_x: crop_x + crop_w] = crop_rgb
        
        
        cv2.imshow('test', rgb)


        file_name = str(idx).zfill(2)
        cv2.imwrite('./image_results/img' + file_name + '.jpg', rgb)
        if cv2.waitKey(750) == ord('q'): # q를 누르면 종료
            
            break