import tensorflow as tf
from utils.load_datasets import DatasetGenerator
from utils.plot_generator import plot_generator
from model.model_builder import base_model
import matplotlib.pyplot as plt
import argparse
import cv2
import numpy as np
import open3d as o3d
from tensorflow.python.saved_model import tag_constants


parser = argparse.ArgumentParser()

# Set Convert to SavedMoel
parser.add_argument("--saved_model_path", type=str,   help="저장된 모델 가중치 경로",
                    default='./checkpoints/0712/_0712_B8_E100_LR-0.001_320-320_train_HRNet_best_loss.h5')
parser.add_argument("--test_img_path", type=str,   help="테스트 할 이미지 경로",
                    default='./test_img2.jpg')
parser.add_argument("--image_size",       type=tuple,  help="조정할 이미지 크기 설정",
                    default=(640, 480))

# Set dataset directory path 
parser.add_argument("--dataset_dir",      type=str,    help="데이터셋 다운로드 디렉토리 설정",
                    default='./datasets/')


args = parser.parse_args()
     

# tf.debugging.set_log_device_placement(True)

model = base_model(image_size=args.image_size, output_channel=1)
model.load_weights(args.saved_model_path)



def project_p3d(p3d, cam_scale, K):
    p3d = p3d * cam_scale
    p2d = np.dot(p3d, K.T)
    p2d_3 = p2d[:, 2]
    p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
    p2d[:, 2] = p2d_3
    p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
    return p2d

def draw_arrow(img, p2ds, thickness=1):
    h, w = img.shape[0], img.shape[1]
    for idx, pt_2d in enumerate(p2ds):
        p2ds[idx, 0] = np.clip(pt_2d[0], 0, w)
        p2ds[idx, 1] = np.clip(pt_2d[1], 0, h)
    
    img = cv2.arrowedLine(img, (p2ds[0,0], p2ds[0,1]), (p2ds[1,0], p2ds[1,1]), (0,0,255), thickness)
    img = cv2.arrowedLine(img, (p2ds[0,0], p2ds[0,1]), (p2ds[2,0], p2ds[2,1]), (0,255,0), thickness)
    img = cv2.arrowedLine(img, (p2ds[0,0], p2ds[0,1]), (p2ds[3,0], p2ds[3,1]), (255,0,0), thickness)
    return img

if __name__ == "__main__":

    # Load TensorRT Model
    converted_model_path = '/home/park/park/Tensorflow-Keras-Realtime-Segmentation/checkpoints/export_path_trt/1/'
    
    print('load_model')
    seg_model = tf.saved_model.load(converted_model_path, tags=[tag_constants.SERVING])
    
    print('infer')
    infer = seg_model.signatures['serving_default']

    in_img = cv2.imread(args.test_img_path)
    in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2RGB)
    
    in_img = tf.cast(in_img, tf.float32)
    img = tf.image.resize(in_img, (args.image_size[0], args.image_size[1]), method=tf.image.ResizeMethod.BILINEAR)
    
    img /= 255.
    img = tf.expand_dims(img, axis=0)


    pred_depth = model.predict(img)
    pred_depth = pred_depth[0]

    seg_img = tf.image.resize(in_img, (args.image_size[0], args.image_size[1]), method=tf.image.ResizeMethod.BILINEAR)
    seg_img /= 255.
    seg_img = tf.expand_dims(seg_img, axis=0)

    pred_seg = infer(seg_img)
    preds = pred_seg['output']

        
    output = tf.argmax(preds, axis=-1)

    seg_output = output[0]
    seg_output = (seg_output.numpy() * 127).astype(np.uint8)

    # Get display area
    contours, _ = cv2.findContours(seg_output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tmp = 0
    display_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > tmp: 
            tmp = area
            display_contours.append(contour)
            
    
    x,y,w,h = cv2.boundingRect(display_contours[0])

    
    pred_depth *= 1000.
    
    pred_depth = np.where(pred_depth<=0, 1., pred_depth)
    
    
    # Convert to RGB-D
    
    numpy_img = img[0].numpy()
    numpy_img *= 255
    numpy_img = numpy_img.astype(np.uint8)
    original_img = numpy_img.copy()
    numpy_img = cv2.cvtColor(numpy_img, cv2.COLOR_RGB2GRAY)
    

        
    plt.imshow(pred_depth)
    plt.show()
    
    # numpy_img = numpy_img[y:y+h, x:x+w]
    # pred_depth = pred_depth[y:y+h, x:x+w]

    # depth_mean = np.mean(pred_depth)
    # pred_depth = np.where(pred_depth>=1, depth_mean, pred_depth)
    

    open3d_rgb = o3d.geometry.Image(numpy_img)
    pred_depth = o3d.geometry.Image(pred_depth)
    

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        open3d_rgb, pred_depth)

    

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    
    # plt.imshow(rgbd_image.depth)
    # plt.show()
    
    xyz_load = np.asarray(pcd.points)
    # print('open3d_rgb', np.asarray(open3d_rgb).shape)
    # print('pred_depth', np.asarray(pred_depth).shape)
    
    
    # print(np_pcd.shape)
    pc = np.zeros((480, 640, 3))
    xyz_load = np.reshape(xyz_load, (480, 640, 3))
    pc[:,:,0] = xyz_load[:, :, 0]
    pc[:,:,1] = xyz_load[:, :, 1]
    pc[:,:,2] = xyz_load[:, :, 2]

    center_x = x + (w//2)
    center_y = y + (h//2)
    
    choose_pc = pc[center_y-1:center_y+1, center_x-1:center_x+1]

    pointCloud_area = np.zeros((480, 640), dtype=np.uint8)
    pointCloud_area = cv2.line(pointCloud_area, (center_x-1, center_y-1), (center_x+1, center_y+1), 255, 10, cv2.LINE_AA)
    choose_pc = pc[np.where(pointCloud_area[:, :]==255)]
                
    choose_pc = choose_pc[~np.isnan(choose_pc[:,2])]

    meanarr, comparr, _ = cv2.PCACompute2(choose_pc, mean=None)

    comparr = -comparr
    if comparr[2, 2] < 0:
        comparr[2, :3] = -comparr[2, :3]
                
    # Target Pose 생성
    target_pose = np.identity(4)
    target_pose[:3,:3] = comparr.T # rotation
    target_pose[:3,3] = meanarr # transration

    
    seg_output = tf.expand_dims(seg_output, axis=-1)

    mask = tf.concat([seg_output, seg_output, seg_output], axis=-1)
    original_img = np.where(mask>=1, 255, original_img)

    
    for i in range(20):
        print(i)
        camera_pose = np.identity(3)
        camera_pose[0, 0] = 322
        camera_pose[1, 1] = 322
        camera_pose[0, 2] = 480 //2 
        camera_pose[1, 2] = 640 //2


        arrow = np.array([[0,0,0],[0.05,0,0],[0,0.05,0],[0,0,0.05]])
        arrow = np.dot(arrow, target_pose[:3, :3].T) + target_pose[:3, 3]
        arrow_p2ds = project_p3d(arrow.copy(), 1.0, camera_pose)


        test = draw_arrow(original_img.copy(), arrow_p2ds, thickness=2)
        
        plt.imshow(test)
        plt.show()





