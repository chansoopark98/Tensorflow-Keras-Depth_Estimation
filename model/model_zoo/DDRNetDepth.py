import tensorflow as tf
import keras.utils.conv_utils as conv_utils

class DDRNetDepth(object):
    def __init__(self, image_size: tuple,
        classifier_activation: str, use_multi_gpu: bool = False):
        self.image_size = image_size
        self.classifier_activation = 'swish'
        self.config = None
        self.use_multi_gpu = use_multi_gpu
        self.MOMENTUM = 0.99
        self.EPSILON = 0.001
        self.activation = 'relu' # self.relu


    def build_model(self, hp=None) -> tf.keras.models.Model:
        from .ddrnet import ddrnet_23_slim
        
        model = ddrnet_23_slim(num_classes=1, input_shape=(*self.image_size, 3))

        # model = tf.keras.models.Model(inputs=input_tensor, outputs=output)
        print('Model parameters => {0}'.format(model.count_params()))
        return model

if __name__ == '__main__':
    model = DDRNetDepth(image_size=(512, 512), classifier_activation=None)
    model.build_model().summary()
    