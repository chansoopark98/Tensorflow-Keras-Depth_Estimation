import pyk4a
from pyk4a import Config, PyK4A
import numpy as np

class PyAzureKinectCamera(object):
    def __init__(self, resolution: str = '720') -> None:
        config = self.__get_camera_config(resolution=resolution)
        self.k4a = PyK4A(config=config)
        self.k4a.start()

        self.capture_buffer = None
    
    def __get_camera_config(self, resolution: str = '720'):
        if resolution == '720':
            color_resolution = pyk4a.ColorResolution.RES_720P
        elif resolution == '1080':
            color_resolution = pyk4a.ColorResolution.RES_1080P
        elif resolution == '1440':
            color_resolution = pyk4a.ColorResolution.RES_1440P
        elif resolution == '2160':
            color_resolution = pyk4a.ColorResolution.RES_2160P
        else:
            raise ValueError('설정한 카메라 해상도를 찾을 수 없습니다.\
                 현재 입력한 해상도 {0}'.format(resolution))

        config = Config(
            color_resolution=color_resolution,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True
        )
        return config

    def get_color_intrinsic_matrix(self) -> np.ndarray:
        return self.k4a._calibration.get_camera_matrix(pyk4a.CalibrationType.COLOR)
    
    def get_depth_intrinsic_matrix(self) -> np.ndarray:
        return self.k4a._calibration.get_camera_matrix(pyk4a.CalibrationType.DEPTH)
        
    def capture(self) -> None:
        self.capture_buffer = self.k4a.get_capture()

    def get_color(self) -> np.ndarray:
        return self.capture_buffer.color
    
    def get_depth(self) -> np.ndarray:
        return self.capture_buffer.depth
    
    def get_pcd(self) -> np.ndarray:
        return self.capture_buffer.depth_point_cloud

    def get_transformed_color(self) -> np.ndarray:
        return self.capture_buffer.transformed_color
        
    def get_transformed_depth(self) -> np.ndarray:
        return self.capture_buffer.transformed_depth
    
    def get_transformed_pcd(self) -> np.ndarray:
        return self.capture_buffer.transformed_depth_point_cloud