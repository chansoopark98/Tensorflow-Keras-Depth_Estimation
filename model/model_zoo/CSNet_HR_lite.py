import tensorflow as tf
from keras.utils import data_utils

class CSNetHRLite(object):
    def __init__(self, image_size: tuple,
        classifier_activation: str, use_multi_gpu: bool = False):
        self.image_size = image_size
        self.classifier_activation = classifier_activation
        self.config = None
        self.use_multi_gpu = use_multi_gpu
        self.MOMENTUM = 0.99
        self.EPSILON = 0.001
        self.activation = self.hard_swish
        self.dropout_rate = 0.2
        self.configuration_default()

    def configuration_default(self):
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out",
                                                  distribution="truncated_normal")
        # Set Sync batch normalization when use multi gpu
        if self.use_multi_gpu:
            self.batch_norm = tf.keras.layers.experimental.SyncBatchNormalization
        else:
            self.batch_norm = tf.keras.layers.BatchNormalization

        # Set model hyper parameters value
        self.config = {
            'stem_units': 16,
            'os2_expand': 1,
            'os2_units': 24,
            'os4_expand': 3,
            'os4_units': 48,
            'os8_expand': 3,
            'os8_units': 56,
            'os16_expand': 6,
            'os16_units': 144,
            'os32_expand': 6,
            'os32_units': 176
        }

    def correct_pad(self, inputs, kernel_size):
        img_dim = 2 if tf.keras.backend.image_data_format() == "channels_first" else 1
        input_size = tf.keras.backend.int_shape(inputs)[img_dim : (img_dim + 2)]
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if input_size[0] is None:
            adjust = (1, 1)
        else:
            adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
        correct = (kernel_size[0] // 2, kernel_size[1] // 2)
        return (
            (correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]),
        )

    def relu6(self, x):
        return tf.keras.layers.ReLU(max_value=6)(x)

    def relu(self, x):
        return tf.keras.layers.ReLU()(x)


    def hard_sigmoid(self, x):
        return tf.keras.layers.ReLU(6.0)(x + 3.0) * (1.0 / 6.0)


    def hard_swish(self, x):
        return tf.keras.layers.Multiply()([x, self.hard_sigmoid(x)])


    def stem_block(self, x: tf.Tensor, filters: int = 16):
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=3,
                                   strides=(2, 2),
                                   padding="same",
                                   use_bias=False,
                                   kernel_initializer=self.kernel_initializer,
                                   name="stem_conv")(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name="stem_bn")(x)
        x = tf.keras.layers.Activation('relu', name='stem_relu')(x)
        return x

    def conv_block(self, x: tf.Tensor, filters: int, expand_ratio: int,
                   kernel_size: int, stride: int, block_id: int):
        shortcut_tensor = x
        input_filters = tf.keras.backend.int_shape(x)[-1]
        x = tf.keras.layers.Conv2D(filters=filters * expand_ratio,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=self.kernel_initializer,
                                   name='expand_1x1_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='expand_bn_{0}'.format(block_id))(x)
        x = tf.keras.layers.Activation(self.activation, name='expand_activation_{0}'.format(block_id))(x)

        if stride == 2:
            x = tf.keras.layers.ZeroPadding2D(padding=self.correct_pad(x, kernel_size),
                                              name='depthwise_pad_{0}'.format(block_id))(x)

        x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                            strides=stride, padding="same" if stride == 1 else "valid",
                                            depthwise_initializer=self.kernel_initializer,
                                            use_bias=False, name='depthwise_conv_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='depthwise_bn_{0}'.format(block_id))(x)
        x = tf.keras.layers.Activation(self.activation, name='depthwise_activation_{0}'.format(block_id))(x)

        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=self.kernel_initializer,
                                   name='conv_1x1_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='conv_bn_{0}'.format(block_id))(x)

        if input_filters == filters and stride == 1:
            # residual connection
            x = tf.keras.layers.Add(name='add_{0}'.format(block_id))([shortcut_tensor, x])
        return x

    def deconv_block(self, x: tf.Tensor, filters: int, expand_ratio: int,
                   kernel_size: int, block_id: int):
        x = tf.keras.layers.Conv2D(filters=filters * expand_ratio,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=self.kernel_initializer,
                                   name='deconv_expand_1x1_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='deconv_expand_bn_{0}'.format(block_id))(x)
        x = tf.keras.layers.Activation(self.activation, name='deconv_expand_activation_{0}'.format(block_id))(x)

        x = tf.keras.layers.UpSampling2D((2, 2), name='deconv_nearest_upsample_{0}'.format(block_id))(x)

        x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                            strides=1, padding="same",
                                            depthwise_initializer=self.kernel_initializer,
                                            use_bias=False, name='deconv_depthwise_conv_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='deconv_depthwise_bn_{0}'.format(block_id))(x)
        x = tf.keras.layers.Activation(self.activation, name='deconv_depthwise_activation_{0}'.format(block_id))(x)

        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=self.kernel_initializer,
                                   name='deconv_conv_1x1_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='deconv_conv_bn_{0}'.format(block_id))(x)
        return x
    
    def fusion_block(self, x: tf.Tensor, skip: tf.Tensor, filters: int,
                   kernel_size: int, block_id: int):
        x = tf.keras.layers.Concatenate(name='fusion_concat_{0}'.format(block_id))([x, skip])
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=self.kernel_initializer,
                                   name='fusion_conv_1x1_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='fusion_bn_{0}'.format(block_id))(x)
        x = tf.keras.layers.Activation(self.activation, name='fusion_activation_{0}'.format(block_id))(x)

        x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                            strides=1, padding="same",
                                            depthwise_initializer=self.kernel_initializer,
                                            use_bias=False, name='fusion_depthwise_conv_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='fusion_depthwise_bn_{0}'.format(block_id))(x)
        x = tf.keras.layers.Activation(self.activation, name='fusion_depthwise_activation_{0}'.format(block_id))(x)

        return x

    def classifier(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, use_bias=True,
                                   name='classifier_conv',
                                   activation=self.classifier_activation,
                                   kernel_initializer=self.kernel_initializer)(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2),
                                         interpolation='bilinear',
                                         name='model_output')(x)
        return x



    def build_model(self, hp=None) -> tf.keras.models.Model:
        # Set configurations
        stem_units = hp.Int("stem_units", min_value=16, max_value=32, step=8) if hp is not None else self.config['stem_units']

        os2_expand = hp.Int("os2_expand", min_value=1, max_value=3, step=1) if hp is not None else self.config['os2_expand']
        os2_units = hp.Int("os2_units", min_value=16, max_value=24, step=8) if hp is not None else self.config['os2_units']

        os4_expand = hp.Int("os4_expand", min_value=1, max_value=3, step=1) if hp is not None else self.config['os4_expand']
        os4_units = hp.Int("os4_units", min_value=32, max_value=48, step=8) if hp is not None else self.config['os4_units']

        os8_expand = hp.Int("os8_expand", min_value=1, max_value=3, step=1) if hp is not None else self.config['os8_expand']
        os8_units = hp.Int("os8_units", min_value=56, max_value=72, step=8) if hp is not None else self.config['os8_units']

        os16_expand = hp.Int("os16_expand", min_value=3, max_value=6, step=1) if hp is not None else self.config['os16_expand']
        os16_units = hp.Int("os16_units", min_value=112, max_value=144, step=8) if hp is not None else self.config['os16_units']

        os32_expand = hp.Int("os32_expand", min_value=3, max_value=6, step=1) if hp is not None else self.config['os32_expand']
        os32_units = hp.Int("os32_units", min_value=144, max_value=176, step=8) if hp is not None else self.config['os32_units']
        

        # 256x256 os1
        input_tensor = tf.keras.Input(shape=(*self.image_size, 3))
        

        # 16 24 32 64 160

        # Stem conv
        stem = self.stem_block(x=input_tensor, filters=stem_units)

        # 128x128 os2
        os2 = self.conv_block(x=stem, filters=os2_units, expand_ratio=os2_expand, kernel_size=3, stride=1, block_id='os2_1')
        os2 = self.conv_block(x=os2, filters=os2_units, expand_ratio=os2_expand, kernel_size=3, stride=1, block_id='os2_output')
        
        # 64x64 os4
        os4 = self.conv_block(x=os2, filters=os4_units, expand_ratio=os4_expand, kernel_size=3, stride=2, block_id='os4_1')
        os4 = self.conv_block(x=os4, filters=os4_units, expand_ratio=os4_expand, kernel_size=3, stride=1, block_id='os4_output')

        # 32x32 os8
        os8 = self.conv_block(x=os4, filters=os8_units, expand_ratio=os8_expand, kernel_size=3, stride=2, block_id='os8_1')
        os8 = self.conv_block(x=os8, filters=os8_units, expand_ratio=os8_expand, kernel_size=3, stride=1, block_id='os8_output')
            
        # 16x16 os16
        os16 = self.conv_block(x=os8, filters=os16_units, expand_ratio=os16_expand, kernel_size=3, stride=2, block_id='os16_1')
        os16 = self.conv_block(x=os16, filters=os16_units, expand_ratio=os16_expand, kernel_size=3, stride=1, block_id='os16_output')
        
        # 8x8 os32
        os32 = self.conv_block(x=os16, filters=os32_units, expand_ratio=os32_expand, kernel_size=3, stride=2, block_id='os32_1')
        os32 = self.conv_block(x=os32, filters=os32_units, expand_ratio=os32_expand, kernel_size=3, stride=1, block_id='os32_output')

        # embedd somethings?

        # os32 decode -> 16x16
        x = self.deconv_block(x=os32,         filters=os16_units, expand_ratio=os16_expand, kernel_size=5, block_id='decode_os32_1')
        x = self.fusion_block(x=x, skip=os16, filters=os16_units, kernel_size=3, block_id='decode_os32_output')
        
        # os16 decode -> 32x32
        x = self.deconv_block(x=x,           filters=os8_units, expand_ratio=os8_expand, kernel_size=5, block_id='decode_os16_1')
        x = self.fusion_block(x=x, skip=os8, filters=os8_units, kernel_size=3, block_id='decode_os16_output')

        # os8 decode -> 64x64
        x = self.deconv_block(x=x,           filters=os4_units, expand_ratio=os4_expand, kernel_size=5, block_id='decode_os8_1')
        x = self.fusion_block(x=x, skip=os4, filters=os4_units, kernel_size=3, block_id='decode_os8_output')

        # os4 decode -> 128x128
        x = self.deconv_block(x=x,           filters=os2_units, expand_ratio=os2_expand, kernel_size=5, block_id='decode_os4_1')
        x = self.fusion_block(x=x, skip=os2, filters=os2_units, kernel_size=3, block_id='decode_os4_output')
       
       # os2 classifer -> 256x256
        x = self.classifier(x=x)

        model = tf.keras.models.Model(inputs=input_tensor, outputs=x)

        return model