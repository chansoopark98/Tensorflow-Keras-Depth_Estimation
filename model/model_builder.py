import tensorflow
from .model_zoo.unet import Unet

def base_model(image_size, output_channel=1):
    model = Unet(image_size=image_size, input_channel=3, output_channel=output_channel)
    

    
    return model