import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import copy
import cv2

if __name__ == "__main__":

    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_1536P,
            depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
            synchronized_images_only=True
        )
    )
    k4a.start()
    
    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    # Capture
    capture = k4a.get_capture()
    rgb = capture.color

    # select roit
    x, y, w, h = cv2.selectROI(rgb)

    # pointcloud roi mask
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 1
    mask = mask.astype(np.int16)
    indicies = np.where(mask==1)
    
    cv2.destroyAllWindows()
    time.sleep(0.5)


    # 배경사이즈를 타겟이미지와 맞춤

    # 트랙바 생성
    panel = np.zeros([100,400], np.uint8)
    cv2.namedWindow('panel')

    def nothing(x):
        pass

    # 트랙바의 범위 설정
    cv2.createTrackbar('L-H', 'panel', 37,179, nothing)
    cv2.createTrackbar('U-H', 'panel', 70,179, nothing)
    
    cv2.createTrackbar('L-S', 'panel', 109,255, nothing)
    cv2.createTrackbar('U-S', 'panel', 255,255, nothing)

    cv2.createTrackbar('L-V', 'panel', 0,255, nothing)
    cv2.createTrackbar('U-V', 'panel', 255,255, nothing)

    
    while cv2.waitKey(100) != ord('q'):
        
        l_h = cv2.getTrackbarPos('L-H', 'panel')
        u_h = cv2.getTrackbarPos('U-H', 'panel')
        l_s = cv2.getTrackbarPos('L-S', 'panel')
        u_s = cv2.getTrackbarPos('U-S', 'panel')
        l_v = cv2.getTrackbarPos('L-V', 'panel')
        u_v = cv2.getTrackbarPos('U-V', 'panel')
        
        lower_green = np.array([l_h,l_s,l_v])
        upper_green = np.array([u_h,u_s,u_v])

        capture = k4a.get_capture()
        rgb = capture.color
        depth = capture.transformed_depth

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb = rgb[:, :, :3].astype(np.uint8)

        object_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

        roi_rgb = rgb.copy()[y:y+h, x:x+w]

        # 크로마키
        
        hsv = cv2.cvtColor(roi_rgb.copy(), cv2.COLOR_BGR2HSV)
        
        green_mask = cv2.inRange(hsv, lower_green, upper_green) # 영상, 최솟값, 최댓값
        
        green_mask = cv2.bitwise_not(green_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # object_mask = cv2.erode(object_mask, kernel, iterations=3)
    
        object_mask[y:y+h, x:x+w] = green_mask
        object_mask = (object_mask / 255.).astype(np.uint16)
        
        depth *= object_mask.astype(np.uint16)
        rgb *= np.expand_dims(object_mask.astype(np.uint8), axis=-1)

        cv2.imshow('panel', rgb)

    cv2.destroyAllWindows()