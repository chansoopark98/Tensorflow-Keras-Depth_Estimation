import tensorflow as tf
from keras.utils import data_utils

class CSNetHRLite(object):
    def __init__(self, image_size: tuple, num_classes: int, pretrained: bool,
        classifier_activation: str, include_top: bool, num_priors: list, use_multi_gpu: bool = False):
        self.image_size = image_size
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.classifier_activation = classifier_activation
        self.num_priors = num_priors
        self.config = None
        self.use_multi_gpu = use_multi_gpu
        self.include_top = include_top
        self.MOMENTUM = 0.99
        self.EPSILON = 0.001
        self.activation = self.relu
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
            'os4_num_layers': 1,
            'os4_units': 48,

            'os8_num_layers': 1,
            'os8_units': 64,

            'os16_num_layers': 2,
            'os16_units': 48,

            'os32_num_layers': 2,
            'os32_units': 96,

            'os64_num_layers': 3,
            'os64_units': 96,

            'os128_num_layers': 3,
            'os128_units': 112,

            'os256_num_layers': 3,
            'os256_units': 144,
        }

        self.detection_config = {
            'head_units': 64,
            'head_layers': 2,
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

        x = tf.keras.layers.UpSampling2D((2, 2), name='deconv_nearest_upsample_{0}'.format(block_id))(x)

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
                                   kernel_initializer=self.kernel_initializer)(x)
        x = tf.keras.layers.Activation(self.classifier_activation, name='classifier_activation')(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2),
                                         interpolation='bilinear',
                                         name='model_output')(x)
        return x



    def build_model(self, hp=None) -> tf.keras.models.Model:
        # Set configurations
        os4_expand = hp.Int("os4_expand", min_value=1, max_value=3, step=1) if hp is not None else self.config['os4_expand']
        os4_units = hp.Int("os4_units", min_value=16, max_value=32, step=8) if hp is not None else self.config['os4_units']

        os8_expand = hp.Int("os8_expand", min_value=1, max_value=3, step=1) if hp is not None else self.config['os8_expand']
        os8_units = hp.Int("os8_units", min_value=32, max_value=48, step=8) if hp is not None else self.config['os8_units']

        os16_expand = hp.Int("os16_expand", min_value=1, max_value=6, step=1) if hp is not None else self.config['os16_expand']
        os16_units = hp.Int("os16_units", min_value=48, max_value=80, step=8) if hp is not None else self.config['os16_units']

        os32_expand = hp.Int("os32_expand", min_value=3, max_value=6, step=1) if hp is not None else self.config['os32_expand']
        os32_units = hp.Int("os32_units", min_value=80, max_value=112, step=32) if hp is not None else self.config['os32_units']
        
        os64_expand = hp.Int("os64_expand", min_value=3, max_value=6, step=1) if hp is not None else self.config['os64_expand']
        os64_units = hp.Int("os64_units", min_value=112, max_value=160, step=16) if hp is not None else self.config['os64_units']

        os128_expand = hp.Int("os128_expand", min_value=3, max_value=6, step=1) if hp is not None else self.config['os128_expand']
        os128_units = hp.Int("os128_units", min_value=112, max_value=160, step=16) if hp is not None else self.config['os128_units']

        os256_expand = hp.Int("os256_expand", min_value=3, max_value=6, step=1) if hp is not None else self.config['os256_expand']
        os256_units = hp.Int("os256_units", min_value=112, max_value=160, step=16) if hp is not None else self.config['os256_units']

        # 256x256 os1
        input_tensor = tf.keras.Input(shape=(*self.image_size, 3))
        

        base_channel = 32 # 16 24 32 64 160

        # Stem conv
        stem = self.stem_block(x=input_tensor, filters=base_channel)

        # 128x128 os2
        os2 = self.conv_block(x=stem, filters=base_channel, expand_ratio=1, kernel_size=3, stride=1, block_id='os2_1')
        os2 = self.conv_block(x=os2, filters=base_channel, expand_ratio=1, kernel_size=3, stride=1, block_id='os2_output')
        
        # 64x64 os4
        os4 = self.conv_block(x=os2, filters=os4_units, expand_ratio=1, kernel_size=3, stride=2, block_id='os4_1')
        os4 = self.conv_block(x=os4, filters=os4_units, expand_ratio=1, kernel_size=3, stride=1, block_id='os4_output')

        # 32x32 os8
        os8 = self.conv_block(x=os4, filters=os8_units, expand_ratio=1, kernel_size=3, stride=2, block_id='os8_1')
        os8 = self.conv_block(x=os8, filters=os8_units, expand_ratio=1, kernel_size=3, stride=1, block_id='os8_output')
            
        # 16x16 os16
        os16 = self.conv_block(x=os8, filters=os16_units, expand_ratio=1, kernel_size=3, stride=2, block_id='os16_1')
        os16 = self.conv_block(x=os16, filters=os16_units, expand_ratio=1, kernel_size=3, stride=1, block_id='os16_output')
        
        # 8x8 os32
        os32 = self.conv_block(x=os16, filters=os32_units, expand_ratio=1, kernel_size=3, stride=2, block_id='os32_1')
        os32 = self.conv_block(x=os32, filters=os32_units, expand_ratio=1, kernel_size=3, stride=1, block_id='os32_output')
        
        # embedd somethings?

        # os32 decode -> 16x16
        x = self.deconv_block(x=os32, filters=160, expand_ratio=3, kernel_size=5, block_id='decode_os32_1')
        x = self.fusion_block(x=x, skip=os16, filters=160, kernel_size=3, stride=1, block_id='decode_os32_output')
        
        # os16 decode -> 32x32
        x = self.deconv_block(x=x, filters=64, expand_ratio=3, kernel_size=5, block_id='decode_os16_1')
        x = self.fusion_block(x=x, skip=os8, filters=64, kernel_size=3, stride=1, block_id='decode_os16_output')

        # os8 decode -> 64x64
        x = self.deconv_block(x=x, filters=32, expand_ratio=3, kernel_size=5, block_id='decode_os8_1')
        x = self.fusion_block(x=x, skip=os4, filters=64, kernel_size=3, stride=1, block_id='decode_os8_output')

        # os4 decode -> 128x128
        x = self.deconv_block(x=x, filters=24, expand_ratio=3, kernel_size=5, block_id='decode_os4_1')
        x = self.fusion_block(x=x, skip=os2, filters=24, kernel_size=3, stride=1, block_id='decode_os4_output')
       
       # os2 classifer -> 256x256
        x = self.classifier(x=x)

        model = tf.keras.models.Model(inputs=input_tensor, outputs=x)

        if self.pretrained:
            # weights_path = data_utils.get_file(
            #     'CSNet-NAS-beta.h5',
            #     'https://github.com/TSPXR/pretrained-weights-db/releases/download/v0.0.6/CSNet-HR-Lite-class6-224x224-b16-lr0.001-ep100-adam-focal-single-fineTuneModel_best_val_loss.h5',
            #     cache_subdir="models"
            # )
            model.summary()
            # model.load_weights(weights_path)

        return model

    def build_extra_layer(self, hp=None) -> tf.keras.models.Model:
        backbone_hp = hp if hp is not None else None
        base = self.build_model(hp=backbone_hp)
        
        model_input = base.input
        
        x1 = base.get_layer('add_os16_output').output # 16x16
        x2 = base.get_layer('add_os32_output').output # 8x8
        x3 = base.get_layer('add_os64_output').output # 4x4
        x4 = base.get_layer('add_os128_output').output # 2x2
        x5 = base.get_layer('add_os256_output').output # 1x1

        head_units = hp.Int("head_units", min_value=64, max_value=160, step=16) if hp is not None else self.detection_config['head_units']
        head_layers = hp.Int("head_layers", min_value=0, max_value=2, step=1) if hp is not None else self.detection_config['head_layers']
        head_kernel = hp.Int("head_kernel", min_value=1, max_value=3, step=2) if hp is not None else self.detection_config['head_kernel']
        
        # Build FPN
        features = [x1, x2, x3, x4, x5]
        # features = list(features)
        mbox_conf = []
        mbox_loc = []

        for i, x in enumerate(features):
            name = x.name.split(':')[0]  # name만 추출 (ex: block3b_add)

            # if normalizations is not None and normalizations[i] > 0:
            #     x = Normalize(normalizations[i], name=name + '_norm')(x)
            
            conf_input = x
            box_input = x

            # classification head
            for cls_idx in range(head_layers):
                conf_input = tf.keras.layers.SeparableConv2D(head_units, 3, padding='same',
                                    depthwise_initializer=self.kernel_initializer,
                                    pointwise_initializer=self.kernel_initializer,
                                    name='{0}_cls_sep_conv_{1}'.format(name, cls_idx),
                                    use_bias=False)(conf_input)
                conf_input = self.batch_norm(epsilon=self.EPSILON,
                                momentum=self.MOMENTUM,
                                name='{0}_cls_sep_bn_{1}'.format(name, cls_idx))(conf_input)
                conf_input = tf.keras.layers.Activation('relu', name='{0}_cls_sep_activation_{1}'.format(name, cls_idx))(conf_input)

            conf_input = tf.keras.layers.SeparableConv2D(self.num_priors[i] * self.num_classes, head_kernel, padding='same',
                                    depthwise_initializer=self.kernel_initializer,
                                    pointwise_initializer=self.kernel_initializer,
                                    name='{0}_cls_sep_classifier'.format(name),
                                    use_bias=True)(conf_input)
            conf_input = tf.keras.layers.Flatten(name='{0}_mbox_conf_flat'.format(name))(conf_input)
            mbox_conf.append(conf_input)

            # localization head
            for loc_idx in range(head_layers):
                box_input = tf.keras.layers.SeparableConv2D(head_units, 3, padding='same',
                                    depthwise_initializer=self.kernel_initializer,
                                    pointwise_initializer=self.kernel_initializer,
                                    name='{0}_loc_sep_conv_{1}'.format(name, loc_idx),
                                    use_bias=False)(box_input)
                box_input = self.batch_norm(epsilon=self.EPSILON,
                                momentum=self.MOMENTUM,
                                name='{0}_loc_sep_bn_{1}'.format(name, loc_idx))(box_input)
                box_input = tf.keras.layers.Activation('relu', name='{0}_loc_sep_activation_{1}'.format(name, loc_idx))(box_input)

            box_input = tf.keras.layers.SeparableConv2D(self.num_priors[i] * 4, head_kernel, padding='same',
                                                depthwise_initializer=self.kernel_initializer,
                                                pointwise_initializer=self.kernel_initializer,
                                                bias_initializer=PriorProbability(probability=0.01),
                                                name='{0}_loc_sep_classifier'.format(name),
                                                use_bias=True)(box_input)
            box_input = tf.keras.layers.Flatten(name='{0}_mbox_loc_flat'.format(name))(box_input)
            mbox_loc.append(box_input)

        mbox_loc = tf.keras.layers.Concatenate(
            axis=1, name='mbox_loc')(mbox_loc)
        mbox_loc = tf.keras.layers.Reshape(
            (-1, 4), name='mbox_loc_final')(mbox_loc)

        mbox_conf = tf.keras.layers.Concatenate(
            axis=1, name='mbox_conf')(mbox_conf)
        mbox_conf = tf.keras.layers.Reshape(
            (-1, self.num_classes), name='mbox_conf_logits')(mbox_conf)
        # mbox_conf = tf.keras.layers.Activation('sigmoid', name='mbox_conf_logits_sigmoid')(mbox_conf)

        # predictions/concat:0, shape=(Batch, 8732, 25)
        predictions = tf.keras.layers.Concatenate(
            axis=2, name='predictions', dtype=tf.float32)([mbox_conf, mbox_loc])

        model = tf.keras.models.Model(inputs=model_input, outputs=predictions)
        return model