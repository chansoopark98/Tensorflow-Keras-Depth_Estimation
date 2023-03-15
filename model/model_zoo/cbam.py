from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Concatenate, Conv2D, Add, Activation
import tensorflow as tf

def cbam_block(cbam_feature, ratio=8):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""
	
	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature)
	return cbam_feature

def channel_attention(input_feature, ratio=8):
	channel_axis = -1
	
	channel = tf.keras.backend.int_shape(input_feature)[channel_axis]
	
	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	
	avg_pool = shared_layer_one(avg_pool)
	
	avg_pool = shared_layer_two(avg_pool)
	
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	
	max_pool = shared_layer_one(max_pool)
	
	max_pool = shared_layer_two(max_pool)
	
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid')(cbam_feature)
	
	return multiply([input_feature, cbam_feature])

def spatial_attention(input_feature):
	kernel_size = 7
	cbam_feature = input_feature
	
	avg_pool = tf.reduce_mean(cbam_feature, axis=3, keepdims=True)
	max_pool = tf.reduce_max(cbam_feature, axis=3, keepdims=True)
	
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)	
	
		
	return multiply([input_feature, cbam_feature])