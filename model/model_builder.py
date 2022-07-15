from re import I
import tensorflow
from .model_zoo.unet import Unet
import tensorflow.keras.models as models
from tensorflow.keras.layers import Input
from .model_zoo.dense_depth import HRNet
from .model_zoo.monoDepth import Depth_Estimation

     
def base_model(image_size, output_channel=1):
    
    model = Unet(image_size=image_size, input_channel=3, output_channel=output_channel).build_generator()
    # model = model.build_generator()
    # model = HRNet(image_size=image_size)

    # model = Depth_Estimation()
    
    return model
