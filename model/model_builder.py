import tensorflow
from model.model import auto_encoder

def base_model(image_size, num_classes=1):

    model_input, model_output = auto_encoder(input_shape=(image_size[0], image_size[1], 3), classes=num_classes)
    # final = tf.keras.Model(model_input, model_output)

    # return tf.keras.Model(model_input, model_output)
    return model_input, model_output