import tensorflow as tf
from keras.utils import data_utils

class ModelBuilder(object):
    def __init__(self, image_size: tuple = (300, 300),
                 use_weight_decay: bool = False, weight_decay: float = 0.00001, is_tunning: bool = False):
        """
        Args:
            image_size         (tuple) : Model input resolution ([H, W])
            num_classes        (int)   : Number of classes to classify 
                                         (must be equal to number of last filters in the model)
            use_weight_decay   (bool)  : Use weight decay.
            weight_decay       (float) : Weight decay value.
        """
        self.image_size = image_size
        self.use_weight_decay = use_weight_decay
        self.weight_decay = weight_decay

        self.MOMENTUM = 0.99
        self.EPSILON = 0.001

        self.kernel_initializer = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out",
                                                  distribution="truncated_normal")
  
    def build_model(self) -> tf.keras.models.Model:
        # from .model_zoo.CSNet_HR_lite import CSNetHRLite
        # model = CSNetHRLite(image_size=self.image_size,
        #                     classifier_activation=None,
        #                     ).build_model()

        # from .model_zoo.EfficientDeepLab import EfficientDeepLab
        # model = EfficientDeepLab(image_size=self.image_size,
        #                     classifier_activation=None,
        #                     ).build_model()

        # from .model_zoo.Unet import unet
        # model_input, model_output = unet(input_shape=(*self.image_size, 3), base_channel=32, output_channel=1, use_logits=False)

        # from .model_zoo.MobileDepth import MobileDepth
        # model = MobileDepth(image_size=self.image_size, classifier_activation=None).build_model()

        from .model_zoo.EfficientDepth import EfficientDepth
        model = EfficientDepth(image_size=self.image_size, classifier_activation=None).build_model()

        weights_path = data_utils.get_file(
            'EfficientDepth_converted_nyu_pretrained.h5',
            'https://github.com/chansoopark98/Tensorflow-Keras-Depth_Estimation/releases/download/v0.0.1/_Bs-32_Ep-30_Lr-0.001_ImSize-480_Opt-adam_multi-gpu_0310_230310_EfficientDepth_nyuDepth_amsgrad-True_best_loss.h5',
            cache_subdir="models"
        )
        model.load_weights(weights_path, by_name=True)
        
        if self.use_weight_decay:
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer.add_loss(lambda layer=layer: tf.keras.regularizers.L2(
                        self.weight_decay)(layer.kernel))
                elif isinstance(layer, tf.keras.layers.SeparableConv2D):
                    layer.add_loss(lambda layer=layer: tf.keras.regularizers.L2(
                        self.weight_decay)(layer.depthwise_kernel))
                    layer.add_loss(lambda layer=layer: tf.keras.regularizers.L2(
                        self.weight_decay)(layer.pointwise_kernel))
                elif isinstance(layer, tf.keras.layers.DepthwiseConv2D):
                    layer.add_loss(lambda layer=layer: tf.keras.regularizers.L2(
                        self.weight_decay)(layer.depthwise_kernel))
        return model