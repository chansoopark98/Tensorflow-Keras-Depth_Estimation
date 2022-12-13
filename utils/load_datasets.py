import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Union
AUTO = tf.data.experimental.AUTOTUNE

class DataLoadHandler(object):
    def __init__(self, data_dir: str, dataset_name: str):
        """
        This class performs pre-process work for each dataset and load tfds.
        Args:
            data_dir     (str)   : Dataset relative path ( default : './datasets/' )
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.__select_dataset()

    def __select_dataset(self):
        self.dataset_list = self.__load_custom_dataset()

    def __load_custom_dataset(self):
        """
            Loads a custom dataset specified by the user.
        """
        self.train_data = tfds.load(name=self.dataset_name, data_dir=self.data_dir, split='train')
        self.valid_data = tfds.load(name=self.dataset_name, data_dir=self.data_dir, split='validation')
        self.test_data = self.valid_data

        if self.dataset_name == 'nyu_depth_v2':
            self.number_train = 47584
            self.number_valid = 654
        else:
            self.number_train = self.train_data.reduce(0, lambda x, _: x + 1).numpy()
            self.number_valid = self.valid_data.reduce(0, lambda x, _: x + 1).numpy()
        self.number_test = self.number_valid

        # Print  dataset meta data
        print("Number of train dataset = {0}".format(self.number_train))
        print("Number of validation dataset = {0}".format(self.number_valid))
        print("Number of test dataset = {0}".format(self.number_test))

class GenerateDatasets(DataLoadHandler):
    def __init__(self, data_dir: str, image_size: tuple, batch_size: int, dataset_name: str, is_tunning: bool = False):
        """
        Args:
            data_dir         (str)    : Dataset relative path (default : './datasets/').
            image_size       (tuple)  : Model input image resolution.
            batch_size       (int)    : Batch size.
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.is_tunning = is_tunning
        super().__init__(data_dir=self.data_dir, dataset_name=self.dataset_name)


    @tf.function
    def preprocess(self, sample) -> Union[tf.Tensor, tf.Tensor]:
        """
        preprocessing image
        :return:
            RGB image(H,W,3), Depth map(H,W,1)
        """
        img = tf.cast(sample['image'], tf.float32)
        depth = sample['depth']

        depth = tf.expand_dims(depth, axis=-1)
        img = tf.image.resize(img, size=(self.image_size[0], self.image_size[1]), method=tf.image.ResizeMethod.BILINEAR)
        depth = tf.image.resize(depth, size=(self.image_size[0], self.image_size[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # normalize image
        img /= 255.

        # normalize depth map
        # depth = tf.expand_dims(depth, axis=-1)
        # depth /= 10

        return (img, depth)
    
    
    @tf.function
    def augmentation(self, image: tf.Tensor, depth: tf.Tensor)-> Union[tf.Tensor, tf.Tensor]:
        if tf.random.uniform([]) > 0.3:
            image = tf.image.random_jpeg_quality(image, 15, 100)
        if tf.random.uniform([]) > 0.1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5) # 랜덤 채도
        if tf.random.uniform([]) > 0.1:
            image = tf.image.random_brightness(image, max_delta=0.005) # 랜덤 밝기
        if tf.random.uniform([]) > 0.1:
            image = tf.image.random_contrast(image, lower=0.2, upper=0.9) # 랜덤 대비
        if tf.random.uniform([]) > 0.1:
            image = tf.image.random_hue(image, max_delta=0.2) # 랜덤 휴 트랜스폼

        return (image, depth)


    def get_trainData(self, train_data):
        train_data = train_data.shuffle(self.batch_size * 64)
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        # train_data = train_data.map(self.augmentation, num_parallel_calls=AUTO)
        train_data = train_data.padded_batch(self.batch_size)
        train_data = train_data.prefetch(AUTO)
        if self.is_tunning is not True:
            train_data = train_data.repeat()
        return train_data


    def get_validData(self, valid_data):
        valid_data = valid_data.map(self.preprocess, num_parallel_calls=AUTO)
        valid_data = valid_data.padded_batch(self.batch_size).prefetch(AUTO)

        return valid_data


    def get_testData(self, test_data):
        test_data = test_data.map(self.preprocess)
        test_data = test_data.batch(self.batch_size).prefetch(AUTO)

        return test_data