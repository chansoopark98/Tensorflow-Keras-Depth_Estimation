import tensorflow as tf
import tensorflow_datasets as tfds
from utils.augmentations import Augmentation
from typing import Union
AUTO = tf.data.experimental.AUTOTUNE

class DataLoadHandler(object):
    def __init__(self, data_dir: str, dataset_name: str, percentage: int = 100):
        """
        This class performs pre-process work for each dataset and load tfds.
        Args:
            data_dir     (str)   : Dataset relative path ( default : './datasets/' )
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.percentage = percentage
        self.__select_dataset()

    def __select_dataset(self):
        self.dataset_list = self.__load_custom_dataset()

    def __load_custom_dataset(self):
        """
            Loads a custom dataset specified by the user.
            NyuConverted : 
                    train : 47584
                    valid : 654
            DiodeDataset :
                    train : 8574
                    valid : 325
            CustomDataset :
                    train : 656
        """
        
        self.nyu_train = tfds.load(name='NyuConverted', data_dir=self.data_dir, split='train[:{0}%]'.format(self.percentage))
        self.nyu_valid = tfds.load(name='NyuConverted', data_dir=self.data_dir, split='validation')

        # self.diode_train = tfds.load(name='DiodeDataset', data_dir=self.data_dir, split='train[:{0}%]'.format(self.percentage))
        # self.diode_valid = tfds.load(name='DiodeDataset', data_dir=self.data_dir, split='validation')

        # self.custom_train = tfds.load(name='CustomDepth', data_dir=self.data_dir, split='train[:{0}%]'.format(90))
        # self.custom_valid = tfds.load(name='CustomDepth', data_dir=self.data_dir, split='train[{0}%:]'.format(90))
        
        self.train_data = self.nyu_train #.concatenate(self.diode_train) # self.nyu_train.concatenate(self.custom_train)
        self.valid_data = self.nyu_valid #.concatenate(self.diode_valid)
        self.test_data = self.valid_data

        # self.number_custom_train = self.custom_train.reduce(0, lambda x, _: x + 1).numpy()
        # self.number_custom_valid = self.custom_valid.reduce(0, lambda x, _: x + 1).numpy()
        
        self.number_train = 47584 # self.train_data.reduce(0, lambda x, _: x + 1).numpy()
        self.number_valid = 654 # self.valid_data.reduce(0, lambda x, _: x + 1).numpy()
        self.number_test = self.number_valid
        

        self.train_data.shuffle(self.number_train)

        # Print  dataset meta data
        print("Number of train dataset = {0}".format(self.number_train))
        print("Number of validation dataset = {0}".format(self.number_valid))
        print("Number of test dataset = {0}".format(self.number_test))

class GenerateDatasets(DataLoadHandler):
    def __init__(self, data_dir: str, image_size: tuple, batch_size: int, dataset_name: str, is_tunning: bool = False, percentage: int = 100):
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
        self.percentage = percentage
        self.augmentations = Augmentation(image_size=self.image_size, max_crop_scale=1.5)
        super().__init__(data_dir=self.data_dir, dataset_name=self.dataset_name, percentage=self.percentage)


    @tf.function
    def preprocess(self, sample) -> Union[tf.Tensor, tf.Tensor]:
        """
        preprocessing image
        :return:
            RGB image(H,W,3), Depth map(H,W,1)
        """
        image = tf.cast(sample['image'], tf.float32)
        depth = tf.cast(sample['depth'], tf.float32)
        depth = tf.expand_dims(depth, axis=-1)

        return (image, depth)

    @tf.function
    def preprocess_valid(self, sample) -> Union[tf.Tensor, tf.Tensor]:
        """
        preprocessing  valid image
        :return:
            RGB image(H,W,3), Depth map(H,W,1)
        """
        image = tf.cast(sample['image'], tf.float32)
        depth = tf.cast(sample['depth'], tf.float32)
        depth = tf.expand_dims(depth, axis=-1)

        image = tf.image.resize(image, size=(self.image_size[0], self.image_size[1]), method=tf.image.ResizeMethod.BILINEAR)
        depth = tf.image.resize(depth, size=(self.image_size[0], self.image_size[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        image, depth = self.augmentations.normalize(image=image, depth=depth)

        return (image, depth)
    
    @tf.function
    def augmentation(self, image: tf.Tensor, depth: tf.Tensor)-> Union[tf.Tensor, tf.Tensor]:
        # Transform augmentation
        # if tf.random.uniform([]) > 0.5:
        #     image, depth = self.augmentations.random_crop(image=image, depth=depth)
        # else:
        image = tf.image.resize(image, size=(self.image_size[0], self.image_size[1]), method=tf.image.ResizeMethod.BILINEAR)
        depth = tf.image.resize(depth, size=(self.image_size[0], self.image_size[1]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        if tf.random.uniform([]) > 0.5:
            image, depth = self.augmentations.random_rotate(image=image, depth=depth)

        # Color augmentation
        if tf.random.uniform([]) > 0.5:
            image, depth = self.augmentations.random_gamma(image=image, depth=depth)
        if tf.random.uniform([]) > 0.5:
            image, depth = self.augmentations.random_brightness(image=image, depth=depth)
        if tf.random.uniform([]) > 0.5:
            image, depth = self.augmentations.random_color(image=image, depth=depth)

        if tf.random.uniform([]) > 0.5:
            image, depth = self.augmentations.horizontal_flip(image=image, depth=depth)

        image, depth = self.augmentations.normalize(image=image, depth=depth)
        return (image, depth)

    def get_trainData(self, train_data: tf.data.Dataset):
        train_data = train_data.shuffle(self.batch_size * 64)
        train_data = train_data.map(self.preprocess, num_parallel_calls=AUTO)
        train_data = train_data.map(self.augmentation, num_parallel_calls=AUTO)
        train_data = train_data.padded_batch(self.batch_size)
        train_data = train_data.prefetch(AUTO)
        if self.is_tunning is not True:
            train_data = train_data.repeat()
        return train_data

    def get_validData(self, valid_data: tf.data.Dataset):
        valid_data = valid_data.map(self.preprocess_valid, num_parallel_calls=AUTO)
        valid_data = valid_data.padded_batch(self.batch_size).prefetch(AUTO)

        return valid_data

    def get_testData(self, test_data: tf.data.Dataset):
        test_data = test_data.map(self.preprocess_valid)
        test_data = test_data.batch(self.batch_size).prefetch(AUTO)

        return test_data