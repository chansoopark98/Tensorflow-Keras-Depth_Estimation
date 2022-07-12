import tensorflow
from .model_zoo.unet import Unet
from .model_zoo.dense_depth import DepthEstimate


def base_model(image_size, output_channel=1):
    
    # model = Unet(image_size=image_size, input_channel=3, output_channel=output_channel)
    # model = model.build_generator()
    model = DepthEstimate()
    return model