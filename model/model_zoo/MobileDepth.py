import tensorflow as tf

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

    def configuration_default(self):
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out",
                                                  distribution="truncated_normal")
        # Set Sync batch normalization when use multi gpu
        self.batch_norm = tf.keras.layers.BatchNormalization


    def ffm_module(self, x: tf.Tensor, skip: tf.Tensor, filters: int,  kernel_size: int, block_id: str):
        # x: 1x1 conv
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   kernel_initializer=self.kernel_initializer,
                                   use_bias=False, name='ffm_x_conv1x1_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='ffm_x_conv1x1_bn_{0}'.format(block_id))(x)
        x = tf.keras.layers.Activation(self.activation, name='ffm_x_activation_{0}'.format(block_id))(x)

        # x: Resize
        size_before = tf.keras.backend.int_shape(skip)
        x = tf.keras.layers.experimental.preprocessing.Resizing(*size_before[1:3],
                                                                 interpolation='nearest',
                                                                 name='ffm_x_resize_{0}'.format(block_id))(x)
        # x: Depth-wise conv
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                            strides=1, padding="same",
                                            depthwise_initializer=self.kernel_initializer,
                                            use_bias=False, name='ffm_x_dconv_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='ffm_x_dconv_bn{0}'.format(block_id))(x)
        x = tf.keras.layers.Activation(self.activation, name='ffm_x_dconv_activation_{0}'.format(block_id))(x)

        # x: after 1x1 conv
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   kernel_initializer=self.kernel_initializer,
                                   use_bias=False, name='ffm_x_after_conv1x1_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='ffm_x_after_conv1x1_bn_{0}'.format(block_id))(x)
        x = tf.keras.layers.Activation(self.activation, name='ffm_x_after_activation_{0}'.format(block_id))(x)

        # concatenate skip layer
        x = tf.keras.layers.Concatenate(axis=-1)([x, skip])

        # x: Depth-wise conv
        x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                            strides=1, padding="same",
                                            depthwise_initializer=self.kernel_initializer,
                                            use_bias=False, name='ffm_x_after_dconv_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='ffm_x_after_dconv_bn_{0}'.format(block_id))(x)
        x = tf.keras.layers.Activation(self.activation, name='ffm_x_after_dconv_activation_{0}'.format(block_id))(x)

        # x: after 1x1 conv
        x = tf.keras.layers.Conv2D(filters=filters,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   kernel_initializer=self.kernel_initializer,
                                   use_bias=False, name='ffm_x_after_conv1x1_2_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='ffm_x_after_conv1x1_bn_2_{0}'.format(block_id))(x)
        x = tf.keras.layers.Activation(self.activation, name='ffm_x_after_activation_2_{0}'.format(block_id))(x)

        return x


    def classifier(self, x: tf.Tensor, upsample: bool) -> tf.Tensor:

        x = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, use_bias=True,
                                    padding='same',
                                   name='classifier_conv',
                                   kernel_initializer=self.kernel_initializer)(x)

        if upsample:
            x = tf.keras.layers.UpSampling2D(size=(2, 2),
                                            name='resized_model_output')(x)

        return x

    def build_model(self, hp=None) -> tf.keras.models.Model:
        # base = tf.keras.applications.MobileNetV3Large(input_shape=(*self.image_size, 3),
        #                                        alpha=1,
        #                                        minimalistic=False,
        #                                        include_top=False,
        #                                        classes=0,
        #                                        include_preprocessing=False)

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
        
        x = self.ffm_module(x=x, skip=skip16, filters=112, kernel_size=5, block_id='x16')
        x = self.ffm_module(x=x, skip=skip8, filters=80, kernel_size=5, block_id='x8')
        x = self.ffm_module(x=x, skip=skip4, filters=40, kernel_size=5, block_id='x4')
        x = self.ffm_module(x=x, skip=skip2, filters=24, kernel_size=5, block_id='x2')

        # 160 / 112 / 80 / 40 / 24 
        # os32 expanded_conv_14/Add
        # os16 expanded_conv_11/Add
        # os8  expanded_conv_5/Add
        # os4  expanded_conv_2/Add
        # os2  expanded_conv/Add
       
       # os2 classifer -> 256x256
        output = self.classifier(x=x, upsample=True)

        model = tf.keras.models.Model(inputs=input_tensor, outputs=output)
        print('Model parameters => {0}'.format(model.count_params()))
        return model