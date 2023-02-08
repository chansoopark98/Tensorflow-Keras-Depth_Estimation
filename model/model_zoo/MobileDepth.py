import tensorflow as tf
import keras.utils.conv_utils as conv_utils

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

class MobileDepth(object):
    def __init__(self, image_size: tuple,
        classifier_activation: str, use_multi_gpu: bool = False):
        self.image_size = image_size
        self.classifier_activation = None
        self.config = None
        self.use_multi_gpu = use_multi_gpu
        self.MOMENTUM = 0.999
        self.EPSILON = 0.001
        self.activation = 'relu' # self.relu
        self.configuration_default()
    
    def hard_sigmoid(self, x):
        return tf.keras.layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)

    def hard_swish(self, x):
        return tf.keras.layers.Multiply()([x, self.hard_sigmoid(x)])

    def configuration_default(self):
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out",
                                                  distribution="truncated_normal")
        # Set Sync batch normalization when use multi gpu
        self.batch_norm = tf.keras.layers.BatchNormalization

    
    def up_project(self, x, skip, filters, prefix):
        "up_project function"
        x = BilinearUpSampling2D((2, 2), name=prefix+'_upsampling2d')(x)

        x = tf.keras.layers.Concatenate(name=prefix+'_concat')([x, skip])
        x = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=prefix+'_convA')(x)
        # x = tf.keras.layers.BatchNormalization(momentum=0.999)(x)
        x = tf.keras.layers.Activation('relu')(x)
        # x = self.hard_swish(x)
        
        x = tf.keras.layers.SeparableConv2D(filters=filters, kernel_size=3, strides=1, padding='same', name=prefix+'_convB')(x)
        x = tf.keras.layers.Activation('relu')(x)
        # x = tf.keras.layers.BatchNormalization(momentum=0.999)(x)
        # x = self.hard_swish(x)
        return x

    def classifier(self, x: tf.Tensor, upsample: bool = True) -> tf.Tensor:
        x = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, use_bias=True,
                                    padding='same',
                                   name='classifier_conv',
                                   kernel_initializer=self.kernel_initializer)(x)
        if upsample:
            x = BilinearUpSampling2D((2, 2), name='final_upsampling2d')(x)
        # x = tf.keras.layers.DepthwiseConv2D(kernel_size=3, strides=1, use_bias=True,
        #                             padding='same',
        #                            name='classifier_dw_conv',
        #                            kernel_initializer=self.kernel_initializer)(x)
        return x

    def build_model(self, hp=None) -> tf.keras.models.Model:
        input_tensor = tf.keras.Input(shape=(*self.image_size, 3))


        base = tf.keras.applications.MobileNetV3Large(
                                        input_shape=(*self.image_size, 3),
                                        alpha=1.0,
                                        minimalistic=False,
                                        include_top=False,
                                        input_tensor=input_tensor,
                                        include_preprocessing=False
        )

        skip2 = base.get_layer('expanded_conv/Add').output
        skip4 = base.get_layer('expanded_conv_2/Add').output
        skip8 = base.get_layer('expanded_conv_5/Add').output
        skip16 = base.get_layer('expanded_conv_11/Add').output
        x = base.get_layer('expanded_conv_14/Add').output
        
        # 160 / 112 / 80 / 40 / 24 / 16
        x = self.up_project(x=x, skip=skip16, filters=112, prefix='os16')
        x = self.up_project(x=x, skip=skip8, filters=96, prefix='os8')
        x = self.up_project(x=x, skip=skip4, filters=48, prefix='os4')
        x = self.up_project(x=x, skip=skip2, filters=32, prefix='os2')
        
        # os2 classifer -> 128x64
        output = self.classifier(x=x)

        model = tf.keras.models.Model(inputs=input_tensor, outputs=output)
        print('Model parameters => {0}'.format(model.count_params()))
        return model