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
                    default='0317/_Bs-8_Ep-30_Lr-0.0002_ImSize-480_Opt-adam_multi-gpu_0317_230317_Test_best_loss.h5')

args = parser.parse_args()


if __name__ == '__main__':
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

    while True:
        camera.capture()
        frame = camera.get_color()
        
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)



        rgb_image = tf.image.resize(rgb_image, size=args.image_size,
                method=tf.image.ResizeMethod.BILINEAR)

        img = tf.cast(rgb_image, tf.float32)
        img /= 255.
        
        img = tf.expand_dims(img, axis=0)

        pred = model.predict(img)


        rgb_image = tf.image.resize(rgb_image, size=(1536, 2048),
                method=tf.image.ResizeMethod.BILINEAR)

        pred = tf.image.resize(pred, size=(1536, 2048),
                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        


        pred = 1000 / pred

        pred = pred[0].numpy()
        

        pred = camera.get_transformed_depth()
        # pred = pred * 1000

        plt.imshow(pred)
        plt.show()

         # rgb image scaling 
        rgb_image = rgb_image.numpy()
        rgb_image = rgb_image.astype('uint8')

        width = rgb_image.shape[1]
        height = rgb_image.shape[0]
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
        # pcd.voxel_down_sample(0.1)

        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        # pcd = pcd.select_by_index(ind)

        o3d.visualization.draw_geometries([pcd])