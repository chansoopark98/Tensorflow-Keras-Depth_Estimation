import tensorflow as tf
from .EfficientNetV2 import *

def get_resnet_features(model: str = 'resnet50',
                        image_size: tuple = (512, 512),
                        pretrained: bool = True) -> list:

    if model == 'resnet50':
        print('Load {0} model'.format(model))
        base = tf.keras.applications.ResNet50V2(include_top=False, input_shape=(*image_size, 3), classes=0, classifier_activation=None)
        base.summary()        
        
        os2 = base.get_layer('conv1_conv').output # @64
        os4 = base.get_layer('conv2_block3_1_relu').output # @64
        os8 = base.get_layer('conv3_block4_1_relu').output # @128
        os16 = base.get_layer('conv4_block6_1_relu').output # @256
        os32 = base.get_layer('conv5_block3_2_relu').output # @512
        

    elif model == 'resnet101':
        print('Load {0} model'.format(model))
        base = tf.keras.applications.ResNet101(include_top=False, input_shape=(*image_size, 3), classes=0, classifier_activation=None)
        base.summary()
                
        os2 = base.get_layer('conv1_relu').output # @64
        os4 = base.get_layer('conv2_block3_out').output # @256
        os8 = base.get_layer('conv3_block4_out').output # @512
        os16 = base.get_layer('conv4_block23_out').output # @1024
        os32 = base.get_layer('conv5_block3_out').output # @2048
    else:
        raise ValueError('Cannot find your model. Resnet series only valid resnet50, resnet101')
    features = [base, os2, os4, os8, os16, os32]

    return features

def get_efficientnetv2_features(model: str = 'b0',
                        image_size: tuple = (512, 512),
                        pretrained: bool = True) -> list:
    
    if model =='b0':
        # base = EfficientNetV2B0(input_shape=(*image_size, 3), num_classes=0, pretrained=None)
        base = tf.keras.applications.EfficientNetV2B0(include_top=False, include_preprocessing=False, input_shape=(*image_size, 3), classes=0, classifier_activation=None)
        base.summary()

        # os2 = base.get_layer('stem_swish').output # @32
        # os4 = base.get_layer('add').output # @32
        # os8 = base.get_layer('add_1').output # @48
        # os16 = base.get_layer('add_7').output # @112
        # os32 = base.get_layer('add_14').output # @192

        os2 = base.get_layer('stem_activation').output # @32
        os4 = base.get_layer('block2b_add').output # @32
        os8 = base.get_layer('block3b_add').output # @48
        os16 = base.get_layer('block5e_add').output # @112
        os32 = base.get_layer('block6h_add').output # @192

    elif model == 's':
        base = EfficientNetV2S(input_shape=(*image_size, 3), num_classes=0, pretrained=None)
        base.summary()        
        
        # EfficientNetV2S 512 / 256 / 128 / 64 / 32 / 16
        os2 = base.get_layer('add_1').output # @24
        os4 = base.get_layer('add_4').output # @48
        os8 = base.get_layer('add_7').output # @64
        os16 = base.get_layer('add_20').output # @160
        os32 = base.get_layer('add_34').output # @256
    else:
        raise ValueError('Cannot find your model.')
    features = [base, os2, os4, os8, os16, os32]

    return features

if __name__ == '__main__':
    # features = get_resnet_features(model='resnet101', image_size=(512, 512), pretrained=True)
    get_efficientnetv2_features(model='b0', image_size=(512, 512), pretrained=True)