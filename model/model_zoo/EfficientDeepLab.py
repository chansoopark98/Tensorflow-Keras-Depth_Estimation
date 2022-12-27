import tensorflow as tf

class EfficientDeepLab(object):
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


    def classifier(self, x: tf.Tensor, upsample: bool) -> tf.Tensor:

        x = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, use_bias=True,
                                    padding='same',
                                   name='classifier_conv',
                                   kernel_initializer=self.kernel_initializer)(x)
        if self.classifier_activation != None:
            x = tf.keras.layers.Activation(self.classifier_activation, name='classifier_activation')(x)

        if upsample:
            x = tf.keras.layers.UpSampling2D(size=(4, 4),
                                            name='resized_model_output')(x)

        return x

    def build_model(self, hp=None) -> tf.keras.models.Model:
        from .EfficientNetV2 import EfficientNetV2S, EfficientNetV2B0
        from .DeepLabV3Plus import deepLabV3Plus
        # base = EfficientNetV2S(input_shape=(*self.image_size, 3), num_classes=0)
        # skip = base.get_layer('add_4').output
        # x = base.get_layer('add_34').output
        base = EfficientNetV2B0(input_shape=(*self.image_size, 3), num_classes=0)
        
        input_tensor = base.input
        
        skip = base.get_layer('add').output
        x = base.get_layer('add_14').output

        output = deepLabV3Plus(features=[skip, x])

       
       # os2 classifer -> 256x256
        output = self.classifier(x=output, upsample=True)

        model = tf.keras.models.Model(inputs=input_tensor, outputs=output)
        print('Model parameters => {0}'.format(model.count_params()))
        return model