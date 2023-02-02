import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pyk4a
from pyk4a import Config, PyK4A
import open3d as o3d
"""
https://excelsior-cjh.tistory.com/167

"""
def tf_pca(x):
    
    x_cen = x - tf.reduce_mean(x, axis=0)
    s, u, v = tf.linalg.svd(x_cen)

    return s, v.numpy()


def numpy2_pca(X: np.ndarray):
    X_cen = X - X.mean(axis=0)  # 평균을 0으로
    U, D, V_t = np.linalg.svd(X_cen)
    return D, V_t


def project_p3d(p3d, cam_scale, K):
    p3d = p3d * cam_scale
    p2d = np.dot(p3d, K.T)
    p2d_3 = p2d[:, 2]
    p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
    p2d[:, 2] = p2d_3
    p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
    return p2d

def draw_arrow(img, p2ds, thickness=3):
    h, w = img.shape[0], img.shape[1]
    for idx, pt_2d in enumerate(p2ds):
        p2ds[idx, 0] = np.clip(pt_2d[0], 0, w)
        p2ds[idx, 1] = np.clip(pt_2d[1], 0, h)
    
    img = cv2.arrowedLine(img, (p2ds[0,0], p2ds[0,1]), (p2ds[1,0], p2ds[1,1]), (0,0,255), thickness)
    img = cv2.arrowedLine(img, (p2ds[0,0], p2ds[0,1]), (p2ds[2,0], p2ds[2,1]), (0,255,0), thickness)
    img = cv2.arrowedLine(img, (p2ds[0,0], p2ds[0,1]), (p2ds[3,0], p2ds[3,1]), (255,0,0), thickness)
    return img

def _draw_transform(img, trans, camera_intrinsic):
    arrow = np.array([[0,0,0],[0.05,0,0],[0,0.05,0],[0,0,0.05]])
    arrow = np.dot(arrow, trans[:3, :3].T) + trans[:3, 3]
    arrow_p2ds = project_p3d(arrow, 10, camera_intrinsic)
    img = draw_arrow(img, arrow_p2ds, thickness=2)
    return img


if __name__ == "__main__":

    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=False
        )
    )
    k4a.start()

    # getters and setters directly get and set on device
    k4a.whitebalance = 4500
    assert k4a.whitebalance == 4500
    k4a.whitebalance = 4510
    assert k4a.whitebalance == 4510

    while True:
        capture = k4a.get_capture()

        # mm -> cm
        
        depth = capture.depth
        h, w = depth.shape

        grid = np.mgrid[0:depth.shape[0],0:depth.shape[1]]
        u, v = grid[0], grid[1]

        center_x = w // 2
        center_y = h // 2
        focal_length = 300.0
        z = depth / 1000
        adjusted_z = z / focal_length
        x = (u - center_x) * adjusted_z
        y = (v - center_y) * adjusted_z
        pcd = np.stack([x,y,z], axis=-1)

        pointCloud = pcd.copy()
        
        pointCloud_area = np.zeros(pointCloud.shape)
        pointCloud_area[center_y-20:center_y+20, center_x] = 255
        pointCloud_area[center_y, center_x-20:center_x+20] = 255
    
        choose_pc = pointCloud[np.where(pointCloud_area[:, :]==255)]

        # print('choose_pc.shape', choose_pc.shape)
        choose_pc= choose_pc.reshape(-1,3)
        choose_pc = choose_pc[~np.isnan(choose_pc[:,2])]
        # print('final choose_pc.shape', choose_pc.shape)
        
        # PCA 통해 주성분 벡터, 평균, 분산
        a = choose_pc.copy()
        b = choose_pc.copy()
        meanarr, comparr = cv2.PCACompute(a, mean=None)
        e, v = tf_pca(b)

        comparr = -comparr
        v = -v

        if comparr[2, 2] < 0:
            
            comparr[2, :3] = -comparr[2, :3]

        # comparr[2, :3] = abs(comparr[2, :3])
            
        # if v[2, 2] < 0:
        #     v[2, :3] = -v[2, :3]

                    
        # Target Pose 생성
        target_pose = np.identity(4)
        target_pose[:3,:3] = comparr.T # rotation
        target_pose[:3,3] = meanarr # transration

        tf_pose = tf.transpose(v).numpy()


        camera_intrinsic = [[focal_length, 0, center_x],
                            [0, focal_length, center_y],
                            [0, 0, 1]]
        camera_intrinsic = np.array(camera_intrinsic)
        
        pcd = _draw_transform(img = pcd, trans=target_pose, camera_intrinsic=camera_intrinsic)

        cv2.line(pcd, (center_x -20, center_y), (center_x + 20, center_y), (255, 255, 255), thickness=2)
        cv2.line(pcd, (center_x, center_y -20), (center_x, center_y + 20), (255, 255, 255), thickness=2)
        print('target pose : ', target_pose)
        cv2.imshow('test', pcd)
        cv2.waitKey(50)