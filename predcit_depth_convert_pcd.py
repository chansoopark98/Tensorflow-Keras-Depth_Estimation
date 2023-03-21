import os
import glob
import cv2
import argparse
import tensorflow as tf
from model.model_builder import ModelBuilder
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from azure_kinect import PyAzureKinectCamera

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
                    default='0320/_Bs-8_Ep-30_Lr-0.0001_ImSize-480_Opt-adam_multi-gpu_0320_resnet50_lossv2_bottle_best_val_loss.h5')

args = parser.parse_args()


if __name__ == '__main__':
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime('%Y_%m_%d_%H_%M_%S')
    save_dir = './360degree_pointclouds/{0}/'.format(current_time)
    save_rgb_dir = save_dir + 'rgb/'
    save_pcd_dir = save_dir + 'pcd/'
    save_mesh_dir = save_dir + 'mesh/'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_rgb_dir, exist_ok=True)
    os.makedirs(save_pcd_dir, exist_ok=True)
    os.makedirs(save_mesh_dir, exist_ok=True)

    image_list = os.path.join(args.image_dir, '*.' + args.image_format)
    image_list = glob.glob(image_list)

    result_dir = args.image_dir + '/results/'
    os.makedirs(result_dir, exist_ok=True)

    camera = PyAzureKinectCamera(resolution='1536')
    camera.capture()
    intrinsic_matrix = camera.get_color_intrinsic_matrix()

    # Set target transforms
    model = ModelBuilder(image_size=args.image_size).build_model()
    model.load_weights(args.checkpoint_dir + args.weight_name)
    model.summary()

    # select roi
    rgb = camera.get_color()
    x, y, w, h = cv2.selectROI(rgb)

    # pointcloud roi mask
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    mask[y:y+h, x:x+w] = 1
    mask = mask.astype(np.int16)
    indicies = np.where(mask==1)
        
    cv2.destroyAllWindows()
    idx = 0
    while cv2.waitKey(1000) != ord('q'):
        camera.capture()
        frame = camera.get_color()

        cv2.imshow('test', frame)
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



        rgb = tf.image.resize(rgb, size=args.image_size,
                method=tf.image.ResizeMethod.BILINEAR)

        img = tf.cast(rgb, tf.float32)
        img /= 255.
        
        img = tf.expand_dims(img, axis=0)

        pred = model.predict(img)


        rgb = tf.image.resize(rgb, size=(1536, 2048),
                method=tf.image.ResizeMethod.BILINEAR)

        pred = tf.image.resize(pred, size=(1536, 2048),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        


        # pred = 1000 / pred
        pred *= 1000
        pred = pred[0].numpy()

        rgb = rgb.numpy().astype(np.uint8)
        print(rgb.shape)
        object_mask = np.zeros(rgb.shape[:2], dtype=np.uint8)

        roi_rgb = rgb.copy()[y:y+h, x:x+w]

        # 크로마키
        hsv = cv2.cvtColor(roi_rgb.copy(), cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (37, 109, 0), (70, 255, 255)) # 영상, 최솟값, 최댓값
        green_mask = cv2.bitwise_not(green_mask)

        object_mask[y:y+h, x:x+w] = green_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        object_mask = cv2.erode(object_mask, kernel, iterations=2)

        object_mask = (object_mask / 255.).astype(np.uint16)
        
        pred *= np.expand_dims(object_mask.astype(np.uint8), axis=-1)
        rgb *= np.expand_dims(object_mask.astype(np.uint8), axis=-1)
        

        # pred = camera.get_transformed_depth()
        # plt.imshow(pred)
        # plt.show()

         # rgb image scaling 
        rgb = rgb
        rgb = rgb.astype('uint8')

        width = rgb.shape[1]
        height = rgb.shape[0]
        # intrinsic_matrix = np.array([[609.98120117, 0., 637.70794678],
        #                             [0., 609.80639648, 367.38693237],
        #                             [0., 0., 1.]])
        
        # intrinsic_matrix = np.array([[970.65313721, 0.,1026.76464844],
        #                              [0., 970.93304443, 775.31921387],
        #                              [0., 0., 1.]])

        intrinsic_matrix = camera.get_color_intrinsic_matrix()
                                                             
        fx = intrinsic_matrix[0, 0]
        fy = intrinsic_matrix[1, 1]
        cx = intrinsic_matrix[0, 2]
        cy = intrinsic_matrix[1, 2]

        # convert rgb image to open3d depth map
        rgb = o3d.geometry.Image(rgb)

        # depth image scaling
        depth_image = pred.astype('uint16')
        
        # convert depth image to open3d depth map
        depth_image = o3d.geometry.Image(depth_image)
        
        # convert to rgbd image
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,
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
        # pcd.voxel_down_sample(0.1)

        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

        # o3d.visualization.draw_geometries([pcd])

        # Save point cloud
        o3d.io.write_point_cloud(save_pcd_dir + 'test_pointcloud_{0}.pcd'.format(idx), pcd)

        # Save rgb image
        # cv2.imwrite(save_rgb_dir + 'test_rgb_{0}.png'.format(idx), save_rgb)
        idx += 1