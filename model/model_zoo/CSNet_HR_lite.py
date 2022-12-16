import tensorflow as tf

class CSNetHRLite(object):
    def __init__(self, image_size: tuple,
        classifier_activation: str, use_multi_gpu: bool = False):
        self.image_size = image_size
        self.classifier_activation = 'relu'
        self.config = None
        self.use_multi_gpu = use_multi_gpu
        self.MOMENTUM = 0.99
        self.EPSILON = 0.001
        self.activation = 'swish' # self.relu
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
            'stage1_kernel': 3,
            'stage1_layers': 2,
            'stage2_kernel': 3,
            'stage2_layers': 4,
            'stage2_expand': 2,
            'stage3_kernel': 3,
            'stage3_layers': 2,
            'stage3_expand': 6,
            'stage4_kernel': 3,
            'stage4_layers': 6,
            'stage4_expand': 4,
            'stage5_layers': 4,
            'stage5_expand': 4
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

    def swish(self, x):
        return tf.keras.activations.swish(x=x)
    
    def leakly_relu(self, x):
        return tf.keras.layers.LeakyReLU(alpha=0.2)(x)

    def hard_swish(self, x):
        return tf.keras.layers.Multiply()([x, self.hard_sigmoid(x)])

    def stem_block(self, x: tf.Tensor, in_filters: int = 16, stride: int = 1):
        x = tf.keras.layers.Conv2D(filters=in_filters,
                                   kernel_size=3,
                                   strides=stride,
                                   padding="same",
                                   use_bias=False,
                                   kernel_initializer=self.kernel_initializer,
                                   name="stem_conv")(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name="stem_bn")(x)
        x = tf.keras.layers.Activation('relu', name='stem_relu')(x)
        return x
    
    def se_block(self, x: tf.Tensor, se_ratio: int, block_id: str):
        h_axis, w_axis = [1, 2]
        filters = x.shape[-1]
        reduction = filters // se_ratio
        se = tf.reduce_mean(x, [h_axis, w_axis], keepdims=True)
        
        se = tf.keras.layers.Conv2D(filters=reduction,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding='same',
                            use_bias=True,
                            kernel_initializer=self.kernel_initializer,
                            name='se_reduce_conv_1x1_{0}'.format(block_id))(se)
        se = tf.keras.layers.Activation(self.activation, name='se_activation_{0}'.format(block_id))(se)

        se = tf.keras.layers.Conv2D(filters=filters,
                    kernel_size=(1, 1),
                    strides=(1, 1),
                    padding='same',
                    use_bias=True,
                    kernel_initializer=self.kernel_initializer,
                    name='se_conv_1x1_{0}'.format(block_id))(se)
        se = tf.keras.layers.Activation('sigmoid', name='se_sigmoid_activation_{0}'.format(block_id))(se)

        output = tf.keras.layers.Multiply(name='se_multiply_{0}'.format(block_id))([x, se])
        return output

    def conv_block(self, x: tf.Tensor, in_filters: int, out_filters: int, expand_ratio: int,
                   kernel_size: int, stride: int, block_id: int, dilation_rate: int = 1, use_se: bool = False):
        shortcut_tensor = x

        x = tf.keras.layers.Conv2D(filters=in_filters * expand_ratio,
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
                                            dilation_rate=(dilation_rate, dilation_rate),
                                            use_bias=False, name='depthwise_conv_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='depthwise_bn_{0}'.format(block_id))(x)
        x = tf.keras.layers.Activation(self.activation, name='depthwise_activation_{0}'.format(block_id))(x)

        if use_se:
            x = self.se_block(x=x, se_ratio=4 * expand_ratio, block_id=block_id)

        x = tf.keras.layers.Conv2D(filters=out_filters,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=self.kernel_initializer,
                                   name='conv_1x1_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='conv_bn_{0}'.format(block_id))(x)

        if in_filters == out_filters and stride == 1:
            # residual connection
            print(block_id, 'residual connection')
            x = tf.keras.layers.Add(name='add_{0}'.format(block_id))([shortcut_tensor, x])
        return x

    def deconv_block(self, x: tf.Tensor, in_filters: int, out_filters: int, expand_ratio: int,
                   kernel_size: int, block_id: int):
        x = tf.keras.layers.Conv2D(filters=in_filters * expand_ratio,
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

        x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size,
                                            strides=1, padding="same",
                                            depthwise_initializer=self.kernel_initializer,
                                            use_bias=False, name='deconv_depthwise_conv_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='deconv_depthwise_bn_{0}'.format(block_id))(x)
        x = tf.keras.layers.Activation(self.activation, name='deconv_depthwise_activation_{0}'.format(block_id))(x)

        x = tf.keras.layers.Conv2D(filters=out_filters,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=self.kernel_initializer,
                                   name='deconv_conv_1x1_{0}'.format(block_id))(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='deconv_conv_bn_{0}'.format(block_id))(x)

        x = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear', name='deconv_bilinear_upsample_{0}'.format(block_id))(x)
        return x
    
    def fusion_block(self, x: tf.Tensor, skip: tf.Tensor, in_filters: int,
                   kernel_size: int, block_id: int):
        skip = tf.keras.layers.Conv2D(filters=in_filters,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=self.kernel_initializer,
                                   name='skip_conv_1x1_{0}'.format(block_id))(skip)
        skip = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='skip_bn_{0}'.format(block_id))(skip)
        skip = tf.keras.layers.Activation(self.activation, name='skip_activation_{0}'.format(block_id))(skip)

        x = tf.keras.layers.Concatenate(name='fusion_concat_{0}'.format(block_id))([x, skip])
        x = tf.keras.layers.Conv2D(filters=in_filters,
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

    def aspp(self, x: tf.Tensor, in_filters: int, out_filters: int, dropout_rate: float) -> tf.Tensor:
        """Image Feature branch"""
        # b4
        b4 = tf.keras.layers.GlobalAveragePooling2D(name='aspp_b4_gap')(x)
        b4_shape = tf.keras.backend.int_shape(b4)
        b4 = tf.keras.layers.Reshape((1, 1, b4_shape[1]), name='aspp_b4_reshape')(b4)
        b4 = tf.keras.layers.Conv2D(filters=in_filters,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=self.kernel_initializer,
                                   name='aspp_b4_conv')(b4)
        b4 = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='aspp_b4_bn')(b4)
        
        b4 = tf.keras.layers.Activation(self.activation, name='aspp_b4_activation')(b4)
        
        size_before = tf.keras.backend.int_shape(x)
        b4 = tf.keras.layers.experimental.preprocessing.Resizing(*size_before[1:3],
                                                                 interpolation='bilinear',
                                                                 name='aspp_b4_resize')(b4)
        
        # b0
        b0 = tf.keras.layers.Conv2D(filters=in_filters,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=self.kernel_initializer,
                                   name='aspp_b0_conv')(x)
        b0 = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='aspp_b0_bn')(b0)
        b0 = tf.keras.layers.Activation(self.activation, name='aspp_b0_activation')(b0)

        b1 = self.conv_block(x=x, in_filters=in_filters, out_filters=out_filters, expand_ratio=1, kernel_size=3, stride=1, block_id='aspp_b1')
        b2 = self.conv_block(x=x, in_filters=in_filters, out_filters=out_filters, expand_ratio=1, kernel_size=5, stride=1, block_id='aspp_b2')
        b3 = self.conv_block(x=x, in_filters=in_filters, out_filters=out_filters, expand_ratio=1, kernel_size=7, stride=1, block_id='aspp_b3')

        # concatenate ASPP branches & project
        x = tf.keras.layers.Concatenate(axis=-1, name='aspp_concat')([b4, b0, b1, b2, b3])
        x = tf.keras.layers.Conv2D(filters=in_filters,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   padding='same',
                                   use_bias=False,
                                   kernel_initializer=self.kernel_initializer,
                                   name='aspp_concat_conv')(x)
        x = self.batch_norm(epsilon=self.EPSILON,
                            momentum=self.MOMENTUM,
                            name='aspp_concat_bn')(x)
        x = tf.keras.layers.Activation(self.activation, name='aspp_concat_activation')(x)
        
        x = tf.keras.layers.Dropout(dropout_rate, name='aspp_concat_dropout')(x)
        return x

    def classifier(self, x: tf.Tensor, upsample: bool) -> tf.Tensor:
        x = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, use_bias=True,
                                   name='model_output',
                                   activation=self.classifier_activation,
                                   kernel_initializer=self.kernel_initializer)(x)
        if upsample:
            x = tf.keras.layers.UpSampling2D(size=(2, 2),
                                            interpolation='bilinear',
                                            name='resized_model_output')(x)
        return x

    def build_model(self, hp=None) -> tf.keras.models.Model:
        # Set configurations
        # stage1 - stage7 (os2-os32)
        stem_units = 32
        
        stage1_kernel = hp.Int("stage1_kernel", min_value=3, max_value=5, step=2) if hp is not None else self.config['stage1_kernel']
        stage1_layers = hp.Int("stage1_layers", min_value=1, max_value=2, step=1) if hp is not None else self.config['stage1_layers']
        stage1_expand = 1
        stage1_units = 16

        stage2_kernel = hp.Int("stage2_kernel", min_value=3, max_value=5, step=2) if hp is not None else self.config['stage2_kernel']
        stage2_layers = hp.Int("stage2_layers", min_value=1, max_value=2, step=1) if hp is not None else self.config['stage2_layers']
        stage2_expand = hp.Int("stage2_expand", min_value=1, max_value=4, step=2) if hp is not None else self.config['stage2_expand']
        stage2_units = 32
        
        stage3_kernel = hp.Int("stage3_kernel", min_value=3, max_value=5, step=2) if hp is not None else self.config['stage3_kernel']
        stage3_layers = hp.Int("stage3_layers", min_value=1, max_value=3, step=1) if hp is not None else self.config['stage3_layers']
        stage3_expand = hp.Int("stage3_expand", min_value=1, max_value=4, step=2) if hp is not None else self.config['stage3_expand']
        stage3_units = 48

        stage4_kernel = hp.Int("stage4_kernel", min_value=3, max_value=5, step=2) if hp is not None else self.config['stage4_kernel']
        stage4_layers = hp.Int("stage4_layers", min_value=2, max_value=4, step=1) if hp is not None else self.config['stage4_layers']
        stage4_expand = hp.Int("stage4_expand", min_value=1, max_value=4, step=2) if hp is not None else self.config['stage4_expand']
        stage4_units = 96

        stage5_kernel = 3
        stage5_layers = hp.Int("stage5_layers", min_value=3, max_value=5, step=1) if hp is not None else self.config['stage5_layers']
        stage5_expand = hp.Int("stage5_expand", min_value=4, max_value=6, step=2) if hp is not None else self.config['stage5_expand']
        stage5_units = 112

        stage6_kernel = 3
        stage6_layers = hp.Int("stage6_layers", min_value=4, max_value=6, step=1) if hp is not None else self.config['stage6_layers']
        stage6_expand = hp.Int("stage6_expand", min_value=4, max_value=6, step=2) if hp is not None else self.config['stage6_expand']
        stage6_units = 192

        aspp_units = 256
        aspp_dropout = hp.Float("aspp_dropout", min_value=0.1, max_value=0.5, step=0.1) if hp is not None else self.config['aspp_dropout']

        # 256x256 os1
        input_tensor = tf.keras.Input(shape=(*self.image_size, 3))
        
        # 16 24 40 80 112 192 320

        # Stem conv
        stem = self.stem_block(x=input_tensor, in_filters=stem_units, stride=1)

        # Stage 1 : 256x256 os1
        stage1 = self.conv_block(x=stem, in_filters=stem_units, out_filters=stage1_units, expand_ratio=stage1_expand, kernel_size=stage1_kernel, stride=1, block_id='stage1_in')
        for i in range(stage1_layers):
            stage1 = self.conv_block(x=stage1, in_filters=stage1_units, out_filters=stage1_units, expand_ratio=stage1_expand, kernel_size=stage1_kernel, stride=1, block_id='stage1_{0}'.format(i+1))
        
        # Stage 2: 128x128 os2
        stage2 = self.conv_block(x=stage1, in_filters=stage1_units, out_filters=stage2_units, expand_ratio=stage2_expand, kernel_size=stage2_kernel, stride=2, block_id='stage2_in')
        for i in range(stage2_layers):
            stage2 = self.conv_block(x=stage2, in_filters=stage2_units, out_filters=stage2_units, expand_ratio=stage2_expand, kernel_size=stage2_kernel, stride=1, block_id='stage2_{0}'.format(i+1))
        
        # Stage 3: 64x64 os4
        stage3 = self.conv_block(x=stage2, in_filters=stage2_units, out_filters=stage3_units, expand_ratio=stage3_expand, kernel_size=stage3_kernel, stride=2, block_id='stage3_in')
        for i in range(stage3_layers):
            stage3 = self.conv_block(x=stage3, in_filters=stage3_units, out_filters=stage3_units, expand_ratio=stage3_expand, kernel_size=stage3_kernel, stride=1, block_id='stage3_{0}'.format(i+1))
        
        # Stage 4: 32x32 os8
        stage4 = self.conv_block(x=stage3, in_filters=stage3_units, out_filters=stage4_units, expand_ratio=stage4_expand, kernel_size=stage4_kernel, stride=2, block_id='stage4_in', use_se=True)
        for i in range(stage4_layers):
            stage4 = self.conv_block(x=stage4, in_filters=stage4_units, out_filters=stage4_units, expand_ratio=stage4_expand, kernel_size=stage4_kernel, stride=1, block_id='stage4_{0}'.format(i+1), use_se=True)

        # Stage 5: 16x16 os16
        stage5 = self.conv_block(x=stage4, in_filters=stage4_units, out_filters=stage5_units, expand_ratio=stage5_expand, kernel_size=stage5_kernel, stride=2, block_id='stage5_in', use_se=True)
        for i in range(stage5_layers):
            stage5 = self.conv_block(x=stage5, in_filters=stage5_units, out_filters=stage5_units, expand_ratio=stage5_expand, kernel_size=stage5_kernel, stride=1, block_id='stage5_{0}'.format(i+1), use_se=True)

        # Stage 6: 8x8 os32
        stage6 = self.conv_block(x=stage5, in_filters=stage5_units, out_filters=stage6_units, expand_ratio=stage6_expand, kernel_size=stage6_kernel, stride=2, block_id='stage6_in', use_se=True)
        for i in range(stage6_layers):
            stage6 = self.conv_block(x=stage6, in_filters=stage6_units, out_filters=stage6_units, expand_ratio=stage6_expand, kernel_size=stage6_kernel, stride=1, block_id='stage6_{0}'.format(i+1), use_se=True)

        # embedding lightweight Atrous Spatial Pyramid Pooling(ASPP)
        # Image Feature branch
        x = self.aspp(x=stage6, in_filters=stage6_units, out_filters=aspp_units, dropout_rate=aspp_dropout)

        # os32 decode -> 16x16 (os16)
        x = self.deconv_block(x=x, in_filters=aspp_units, out_filters=stage5_units, expand_ratio=1, kernel_size=3, block_id='decode_os32_deconv')
        x = self.fusion_block(x=x, skip=stage5, in_filters=stage5_units, kernel_size=3, block_id='decode_os32_fusion')
        x = self.conv_block(x=x, in_filters=stage5_units, out_filters=stage5_units, expand_ratio=1, kernel_size=3, stride=1, block_id='decode_os32_output')

        # os16 decode -> 32x32 (os8)
        x = self.deconv_block(x=x, in_filters=stage5_units, out_filters=stage4_units, expand_ratio=1, kernel_size=3, block_id='decode_os16_deconv')
        x = self.fusion_block(x=x, skip=stage4, in_filters=stage4_units, kernel_size=3, block_id='decode_os16_fusion')
        x = self.conv_block(x=x, in_filters=stage4_units, out_filters=stage4_units, expand_ratio=1, kernel_size=3, stride=1, block_id='decode_os16_output')

        # os8 decode -> 64x64 (os4)
        x = self.deconv_block(x=x, in_filters=stage4_units, out_filters=stage3_units, expand_ratio=1, kernel_size=3, block_id='decode_os8_deconv')
        x = self.fusion_block(x=x, skip=stage3, in_filters=stage3_units, kernel_size=3, block_id='decode_os8_fusion')
        x = self.conv_block(x=x, in_filters=stage3_units, out_filters=stage3_units, expand_ratio=1, kernel_size=3, stride=1, block_id='decode_os8_output')

        # os4 decode -> 128x128 (os2)
        x = self.deconv_block(x=x, in_filters=stage3_units, out_filters=stage2_units, expand_ratio=1, kernel_size=3, block_id='decode_os4_deconv')
        x = self.fusion_block(x=x, skip=stage2, in_filters=stage2_units, kernel_size=3, block_id='decode_os4_fusion')
        x = self.conv_block(x=x, in_filters=stage2_units, out_filters=stage2_units, expand_ratio=1, kernel_size=3, stride=1, block_id='decode_os4_output')

        # os2 decode -> 256x256 (os1)
        x = self.deconv_block(x=x, in_filters=stage2_units, out_filters=stage1_units, expand_ratio=1, kernel_size=3, block_id='decode_os2_deconv')
        x = self.fusion_block(x=x, skip=stage1, in_filters=stage1_units, kernel_size=3, block_id='decode_os2_fusion')
        x = self.conv_block(x=x, in_filters=stage1_units, out_filters=stage1_units, expand_ratio=1, kernel_size=3, stride=1, block_id='decode_os2_output')
       
       # os2 classifer -> 256x256
        x = self.classifier(x=x, upsample=False)

        model = tf.keras.models.Model(inputs=input_tensor, outputs=x)
        
        print('Model parameters => {0}'.format(model.count_params()))
        return model