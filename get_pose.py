import cv2
import numpy as np
from math import cos, sin
from pyk4a import PyK4A
# 6.123234e-17
x_center = np.array([0, 0])
x_points = np.array([1, 0.05])
y_points = np.array([1, 0.06])

ba = x_points - x_center
bc = y_points - x_center

# def unit_vector(vector):
#     """ Returns the unit vector of the vector.  """
#     return vector / np.linalg.norm(vector)

# def angle_between(v1, v2):
#     """ Returns the angle in radians between vectors 'v1' and 'v2'::

#             >>> angle_between((1, 0, 0), (0, 1, 0))
#             1.5707963267948966
#             >>> angle_between((1, 0, 0), (1, 0, 0))
#             0.0
#             >>> angle_between((1, 0, 0), (-1, 0, 0))
#             3.141592653589793
#     """
#     v1_u = unit_vector(v1)
#     v2_u = unit_vector(v2)
#     return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# def angle_between2(p1, p2):
#     ang1 = np.arctan2(*p1[::-1])
#     ang2 = np.arctan2(*p2[::-1])
#     return np.rad2deg((ang1 - ang2) % (2 * np.pi))

# def AngleBtw2Vectors(vecA, vecB):
#     unitVecA = vecA / np.linalg.norm(vecA)
#     unitVecB = vecB / np.linalg.norm(vecB)
#     return np.arccos(np.dot(unitVecA, unitVecB))

# angle = angle_between(x_center, x_points)
# angle2 = angle_between2(x_center, x_points)

# theta = np.deg2rad(45)
# rot = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
# v2 = np.dot(rot, x_center)

angle = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
print(angle)

# Load camera with the default config
from pyk4a import Config, PyK4A
import pyk4a
k4a = PyK4A()

k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.OFF,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=False,
        )
    )
k4a.start()


# Get the next capture (blocking function)
capture = k4a.get_capture()

img_color = capture.color
depth = capture.depth

# Display with pyplot
from matplotlib import pyplot as plt
while True:
    if np.any(capture.depth):

        plt.imshow(depth/10000) # BGRA to RGB
        plt.show()

""" 두 벡터에서 앵글 계산하는 방법"""
# https://www.youtube.com/watch?v=TTom8n3FFCw

""" OpenCV Contour get angle"""
# https://stackoverflow.com/questions/63971297/how-to-project-pixels-on-eigenvectors-in-opencv