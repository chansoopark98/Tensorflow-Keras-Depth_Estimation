import tensorflow as tf

class TensorBoardImage(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, image, noisy_image):
        super().__init__()
        self.log_dir = log_dir
        self.image = image
        self.noisy_image = noisy_image

    def set_model(self, model):
        self.model = model
        self.writer = tf.summary.create_file_writer(self.log_dir, filename_suffix='images')

    def on_train_begin(self, _):
        self.write_image(self.image, 'Original Image', 0)

    def on_train_end(self, _):
        self.writer.close()

    def write_image(self, image, tag, epoch):
        image_to_write = np.copy(image)
        image_to_write -= image_to_write.min()
        image_to_write /= image_to_write.max()
        with self.writer.as_default():
            tf.summary.image(tag, image_to_write, step=epoch)

    def on_epoch_end(self, epoch, logs={}):
        denoised_image = self.model.predict_on_batch(self.noisy_image)
        self.write_image(denoised_image, 'Denoised Image', epoch)