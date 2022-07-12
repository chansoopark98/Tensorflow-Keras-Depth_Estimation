import tensorflow as tf
import tensorflow_datasets as tfds
AUTO = tf.data.experimental.AUTOTUNE


class DatasetGenerator:
    def __init__(self, data_dir, image_size, batch_size):
        """
        Args:
            data_dir: 데이터셋 상대 경로 ( default : './datasets/' )
            image_size: 백본에 따른 이미지 해상도 크기
            batch_size: 배치 사이즈 크기
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.train_data, self.valid_data = self.initial_load()
        self.minDepth = 10
        self.maxDepth = 1000
        # self.number_train = 47584
        # self.number_valid = 654

    def initial_load(self):
        """
        초기 Tensorflow dataset 로드
        :return:
            train data, validation data
        """
        train_data = tfds.load(name='nyu_depth_v2', data_dir=self.data_dir, split='train')
        valid_data = tfds.load(name='nyu_depth_v2', data_dir=self.data_dir, split='validation')

        self.number_valid = valid_data.reduce(0, lambda x, _: x + 1).numpy()
        self.number_train = train_data.reduce(0, lambda x, _: x + 1).numpy()

        return train_data, valid_data


    def test_preprocess(self, sample):
        img = sample['image']
        depth = sample['depth']
        depth = self.maxDepth / depth
    
        depth = 1. - (tf.clip_by_value(depth, self.minDepth, self.maxDepth) / self.maxDepth)

        return img, depth

    def preprocess(self, sample):
        """
        preprocessing image
        :return:
            RGB image(H,W,3), Depth map(H,W,1)
        """
        img = tf.cast(sample['image'], tf.float32)
        depth = sample['depth']

        depth = tf.expand_dims(depth, axis=-1)

        # Format
        depth = 1. - (tf.clip_by_value(depth, self.minDepth, self.maxDepth) / self.maxDepth)

        return (img, depth)



    def get_trainData(self):
        """
        Set training dataset iterator
        :return:
            train data
        """
        self.train_data = self.train_data.shuffle(1024, reshuffle_each_iteration=True)
        self.train_data = self.train_data.repeat()
        self.train_data = self.train_data.map(self.preprocess, num_parallel_calls=AUTO)
        self.train_data = self.train_data.padded_batch(self.batch_size)
        self.train_data = self.train_data.prefetch(AUTO)

        return self.train_data

    def get_validData(self):
        """
        Set validation dataset iterator
        :return:
            validation data
        """
        self.valid_data = self.valid_data.map(self.preprocess, num_parallel_calls=AUTO)
        self.valid_data = self.valid_data.padded_batch(self.batch_size).prefetch(AUTO)

        return self.valid_data

    def get_testData(self):
        """
        Set test dataset ioterator
        :return:
            test data
        """
        self.test_data = self.valid_data.map(self.test_preprocess, num_parallel_calls=AUTO)
        self.test_data = self.test_data.padded_batch(self.batch_size).prefetch(AUTO)

        return self.test_data