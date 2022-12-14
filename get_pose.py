import numpy as np
import cv2
choose_pc = np.ones([64, 64])

# eigenvalues, eigenvectors = np.linalg.eig(M)
choose_pc = choose_pc[~np.isnan(choose_pc[:,2])]
meanarr, comparr, vararr = cv2.PCACompute2(choose_pc, mean=None)
# meanarr, comparr, vararr = cv2.PCACompute(choose_pc, mean=None)

vararr = vararr.reshape(-1)

comparr = -comparr

# comparr[1, :3] = np.cross(comparr[2, :3], comparr[0, :3])

# 낮은 Depth Quality 로 vector가 튀는걸 보정해주기 위해 수직방향 벡터를 일정 비율로 더해줌
# CORRECTION_RATE = 1.2   # 원래 벡터의 비율
# corrected = CORRECTION_RATE * comparr[2, :3] + (1 - CORRECTION_RATE) * np.array([0,0,1], dtype=np.int8)
# corrected /= np.linalg.norm(corrected)

# comparr[2, :3] = corrected
if comparr[2, 2] < 0:
    comparr[2, :3] = -comparr[2, :3]

# Target Pose 생성
target_pose = np.identity(4)
target_pose[:3,:3] = comparr.T # rotation
target_pose[:3,3] = meanarr # transration

print(target_pose)