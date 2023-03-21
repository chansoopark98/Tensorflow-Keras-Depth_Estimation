import matplotlib.pyplot as plt
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
import time
import numpy as np
import copy
import cv2
from typing import Optional, Tuple

def colorize(image: np.ndarray, clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img


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

    
    while cv2.waitKey(100) != ord('q'):
        
        capture = k4a.get_capture()
        rgb = capture.color
        depth = capture.transformed_depth

        depth = colorize(depth, (0, 1000), colormap=cv2.COLORMAP_JET)
        cv2.imshow('panel', depth)

    cv2.destroyAllWindows()