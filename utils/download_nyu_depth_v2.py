import tensorflow_datasets as tfds

if __name__ == '__main__':
    tfds.load(name='nyu_depth_v2', data_dir='../datasets/', split='train')
