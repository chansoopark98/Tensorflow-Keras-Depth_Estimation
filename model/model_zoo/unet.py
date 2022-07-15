from tensorflow.keras.layers import Conv2D, Add, BatchNormalization, Activation, UpSampling2D, Concatenate, LeakyReLU, Input, Conv2DTranspose, Dropout
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model

class Unet():
    def __init__(self, image_size: tuple, input_channel: int, output_channel: int):
        self.image_size = image_size
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_weights_init = RandomNormal(stddev=0.02)
    
    def build_generator(self):

        gen_input_shape=(self.image_size[0], self.image_size[1], self.input_channel)
        input_src_image = Input(gen_input_shape)

        # encoder model
        e1 = self._encoder_block(input_src_image, 32, batchnorm=False) # 256 1/2
        e1 = self._encoder_block(e1, 32, strides=1)

        e2 = self._encoder_block(e1, 64) # 128 1/4
        e2 = self._encoder_block(e2, 64, strides=1)

        e3 = self._encoder_block(e2, 128) # 64 1/8
        e3 = self._encoder_block(e3, 128, strides=1)

        e4 = self._encoder_block(e3, 256) # 32 1/16
        e4 = self._encoder_block(e4, 256, strides=1)

        e5 = self._encoder_block(e4, 512) # 16 1/32
        e5 = self._encoder_block(e5, 512, strides=1)

        
        # bottleneck, no batch norm and relu
        b = Conv2D(512, (4,4), strides=(2,2), padding='same', use_bias=True, kernel_initializer=self.kernel_weights_init)(e5)
        b = Activation('relu')(b)

        # decoder model
        d3 = self._decoder_block(b, e5, 512)
        d3 = self._encoder_block(d3, 512, strides=1)

        d4 = self._decoder_block(d3, e4, 256, dropout=False)
        d4 = self._encoder_block(d4, 256, strides=1)

        d5 = self._decoder_block(d4, e3, 128, dropout=False)
        d5 = self._encoder_block(d5, 128, strides=1)

        d6 = self._decoder_block(d5, e2, 64, dropout=False)
        d6 = self._encoder_block(d6, 64, strides=1)

        d7 = self._decoder_block(d6, e1, 32, dropout=False)
        d7 = self._encoder_block(d7, 32, strides=1)

        # output
        # g = UpSampling2D()(d7)
        # g = Conv2D(filters=2, kernel_size=4, strides=1, padding='same', kernel_initializer=self.kernel_weights_init)(g)
        
        output = Conv2DTranspose(self.output_channel, (3,3), strides=(2,2), padding='same', kernel_initializer=self.kernel_weights_init)(d7)
    
        # out_image = Activation('tanh')(g)
        
        # define model
        model = Model(inputs=input_src_image, outputs=output, name='generator_model')
        model.trainable = True

        return model

    def _decoder_block(self, layer_in, skip_in, n_filters, strides=(2), dropout=True):
        # add upsampling layer
        g = Conv2DTranspose(n_filters, (3, 3), strides=(strides,strides), padding='same', use_bias=False, kernel_initializer=self.kernel_weights_init)(layer_in)
        # add batch normalization
        g = BatchNormalization()(g, training=True)
        if dropout:
            g = Dropout(0.5)(g, training=True)
        # merge with skip connection
        g = Concatenate()([g, skip_in])
        # relu activation
        g = LeakyReLU(0.2)(g)

        return g

    def _encoder_block(self, layer_in, n_filters, strides=2, batchnorm=True):
        if batchnorm == True:
            use_bias = False
        else:
            use_bias = True
        
        # add downsampling layer
        g = Conv2D(n_filters, (3, 3), strides=(strides, strides), padding='same', use_bias=use_bias, kernel_initializer=self.kernel_weights_init)(layer_in)
        if batchnorm:
            g = BatchNormalization()(g, training=True)
        # leaky relu activation
        g = LeakyReLU(alpha=0.2)(g)

        return g