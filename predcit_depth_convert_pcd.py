import os
import glob
import cv2
import argparse
import tensorflow as tf
from model.model_builder import ModelBuilder
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size",          type=int,    help="Evaluation batch size",
                    default=1)
parser.add_argument("--image_dir",           type=str,    help="Image directory",
                    default='./inputs/')
parser.add_argument("--image_format",           type=str,    help="Image data format (e.g. jpg)",
                    default='png')
parser.add_argument("--image_size",          type=tuple,  help="Model image size (input resolution)",
                    default=(480, 640))
parser.add_argument("--threshold",           type=float,  help="Post processing confidence threshold",
                    default=0.5)
parser.add_argument("--checkpoint_dir",      type=str,    help="Setting the model storage directory",
                    default='./checkpoints/')
parser.add_argument("--weight_name",         type=str,    help="Saved model weights directory",
                    default='0310/_Bs-8_Ep-30_Lr-0.001_ImSize-480_Opt-adam_multi-gpu_0310_230310_EfficientDepth_custom_best_ssim.h5')

args = parser.parse_args()


if __name__ == '__main__':
    image_list = os.path.join(args.image_dir, '*.' + args.image_format)
    image_list = glob.glob(image_list)

    result_dir = args.image_dir + '/results/'
    os.makedirs(result_dir, exist_ok=True)

    # Set target transforms
    model = ModelBuilder(image_size=args.image_size).build_model()
    model.load_weights(args.checkpoint_dir + args.weight_name)
    model.summary()

    cap = cv2.VideoCapture(0)

    # 프레임을 정수형으로 형 변환
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))	# 영상의 넓이(가로) 프레임
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))	# 영상의 높이(세로) 프레임
    
    frame_size = (frameWidth, frameHeight)
    print('frame_size={}'.format(frame_size))

    frameRate = 33

    
    while True:
        # 한 장의 이미지(frame)를 가져오기
        # 영상 : 이미지(프레임)의 연속
        # 정상적으로 읽어왔는지 -> retval
        # 읽어온 프레임 -> frame
        retval, frame = cap.read()
        if not(retval):	# 프레임정보를 정상적으로 읽지 못하면
            break  # while문을 빠져나가기
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        rgb_image = tf.image.resize(rgb_image, size=args.image_size,
                method=tf.image.ResizeMethod.BILINEAR)

        img = tf.cast(rgb_image, tf.float32)
        img /= 255.
        
        img = tf.expand_dims(img, axis=0)

        pred = model.predict(img)


        rgb_image = tf.image.resize(rgb_image, size=(720, 1280),
                method=tf.image.ResizeMethod.BILINEAR)

        pred = tf.image.resize(pred, size=(720, 1280),
                method=tf.image.ResizeMethod.BILINEAR)
        
        pred = pred[0].numpy()
        
        pred = pred * 1000


         # rgb image scaling 
        rgb_image = rgb_image.numpy()
        rgb_image = rgb_image.astype('uint8')

        width = rgb_image.shape[1]
        height = rgb_image.shape[0]
        intrinsic_matrix = np.array([[609.98120117, 0., 637.70794678],
                                    [0., 609.80639648, 367.38693237],
                                    [0., 0., 1.]])
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        # convert rgb image to open3d depth map
        rgb_image = o3d.geometry.Image(rgb_image)

        # depth image scaling
        depth_image = pred.astype('uint16')
        
        # convert depth image to open3d depth map
        depth_image = o3d.geometry.Image(depth_image)
        
        # convert to rgbd image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_image,
                                                                        depth_image,
                                                                        convert_rgb_to_intensity=False)

        test_rgbd_image = np.asarray(rgbd_image)

        print('rgbd shape', test_rgbd_image.shape)
    



        # Create Open3D camera intrinsic object
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width=width,
                                                            height=height,
                                                            fx=fx,
                                                            fy=fy,
                                                            cx=cx,
                                                            cy=cy)
        
        # rgbd image convert to pointcloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)

        o3d.visualization.draw_geometries([pcd])