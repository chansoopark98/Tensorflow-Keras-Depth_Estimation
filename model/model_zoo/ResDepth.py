import tensorflow as tf
import keras.utils.conv_utils as conv_utils
from .cbam import cbam_block

def normalize_data_format(value):
    if value is None:
        value = tf.keras.backend.image_data_format()
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format

class BilinearUpSampling2D(tf.keras.layers.Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(BilinearUpSampling2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = tf.keras.layers.InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):
        input_shape = tf.keras.backend.shape(inputs)
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        
        return tf.image.resize(inputs, [height, width], method=tf.image.ResizeMethod.BILINEAR)

    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
class NearestSampling2D(tf.keras.layers.Layer):
    def __init__(self, size=(2, 2), data_format=None, **kwargs):
        super(NearestSampling2D, self).__init__(**kwargs)
        self.data_format = normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, 'size')
        self.input_spec = tf.keras.layers.InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    height,
                    width)
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
            return (input_shape[0],
                    height,
                    width,
                    input_shape[3])

    def call(self, inputs):
        input_shape = tf.keras.backend.shape(inputs)
        if self.data_format == 'channels_first':
            height = self.size[0] * input_shape[2] if input_shape[2] is not None else None
            width = self.size[1] * input_shape[3] if input_shape[3] is not None else None
        elif self.data_format == 'channels_last':
            height = self.size[0] * input_shape[1] if input_shape[1] is not None else None
            width = self.size[1] * input_shape[2] if input_shape[2] is not None else None
        
        return tf.image.resize(inputs, [height, width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    def get_config(self):
        config = {'size': self.size, 'data_format': self.data_format}
        base_config = super(BilinearUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ResDepth(object):
    def __init__(self, image_size: tuple,
        classifier_activation: str, use_multi_gpu: bool = False):
        self.image_size = image_size
        self.classifier_activation = 'relu'
        self.config = None
        self.use_multi_gpu = use_multi_gpu
        self.MOMENTUM = 0.99
        self.EPSILON = 0.001
        self.activation = 'relu' # self.relu
        self.configuration_default()

    def configuration_default(self):
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out",
                                                  distribution="truncated_normal")
        # Set Sync batch normalization when use multi gpu
        self.batch_norm = tf.keras.layers.BatchNormalization

    # def concat_project(self, x, skip, filters, prefix):
        # x = tf.keras.layers.Concatenate(name=prefix+'_concat')([x, skip])

    def up_project(self, x, skip, filters, prefix):
        "up_project function"
        # x = BilinearUpSampling2D((2, 2), name=prefix+'_bilinear_upsampling2d')(x)
        x = NearestSampling2D((2, 2), name=prefix+'_nearest_upsampling2d')(x)

        skip = cbam_block(skip)
        x = tf.keras.layers.Concatenate(name=prefix+'_concat')([x, skip])

        # Final std conv
        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', use_bias=False, name=prefix+'_conv_1')(x)
        x = tf.keras.layers.BatchNormalization(momentum=self.MOMENTUM)(x)
        x = tf.keras.layers.Activation(self.activation)(x)

        x = tf.keras.layers.Conv2D(filters=filters, kernel_size=3, strides=1, padding='same', use_bias=True, name=prefix+'_conv_2')(x)
        x = tf.keras.layers.Activation(self.activation)(x)

        x = cbam_block(x)
        return x

    def classifier(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, use_bias=True,
                                    padding='same',
                                   name='classifier_conv',
                                   kernel_initializer=self.kernel_initializer)(x)
        
        x = NearestSampling2D((2, 2), name='final_upsampling2d')(x)
       
        return x

    def build_model(self, hp=None) -> tf.keras.models.Model:
        
        # from .EfficientNetV2 import EfficientNetV2S
        # base = EfficientNetV2S(input_shape=(*self.image_size, 3), num_classes=0, pretrained=None)

        base = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(*self.image_size, 3), classes=0, classifier_activation=None)
        # base.summary()        
        input_tensor = base.input
        
        # EfficientNetV2S 512 / 256 / 128 / 64 / 32 / 16
        os2 = base.get_layer('conv1_conv').output # @64
        os4 = base.get_layer('conv2_block3_1_relu').output # @64
        os8 = base.get_layer('conv3_block4_1_relu').output # @128
        os16 = base.get_layer('conv4_block6_1_relu').output # @256
        x = base.get_layer('conv5_block3_2_relu').output # @512
        
        x = self.up_project(x=x, skip=os16, filters=256, prefix='os16')
        x = self.up_project(x=x, skip=os8, filters=128, prefix='os8')
        x = self.up_project(x=x, skip=os4, filters=64, prefix='os4')
        x = self.up_project(x=x, skip=os2, filters=32, prefix='os2')
        
        # os2 classifer -> 256x256
        output = self.classifier(x=x)
        
        model = tf.keras.models.Model(inputs=input_tensor, outputs=output)
        print('Model parameters => {0}'.format(model.count_params()))
        return model

if __name__ == '__main__':
    model = ResDepth(image_size=(512, 512), classifier_activation=None)
    model.build_model()