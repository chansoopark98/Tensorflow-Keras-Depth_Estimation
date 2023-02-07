import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os
def depth_inpaint(depth, max_value=10, missing_value=0):
    depth = np.where(depth > 10, 0, depth)

    depth = cv2.copyMakeBorder(depth, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    mask = (depth == missing_value).astype(np.uint8)

    scale = np.abs(depth).max()
    depth = depth.astype(np.float32) / scale
    depth = cv2.inpaint(depth, mask, 1, cv2.INPAINT_NS)

    depth = depth[1:-1, 1:-1]
    depth = depth * scale

    return depth

dataset_name = 'nyu_depth_v2'
data_dir = './datasets/'
train_data = tfds.load(name=dataset_name, data_dir=data_dir, split='train')
valid_data = tfds.load(name=dataset_name, data_dir=data_dir, split='validation')

train_save_path = '/home/park/nyu_converted/train/'
os.makedirs(train_save_path, exist_ok=True)
os.makedirs(train_save_path + 'image', exist_ok=True)
os.makedirs(train_save_path + 'depth', exist_ok=True)

valid_save_path = '/home/park/nyu_converted/validation/'
os.makedirs(valid_save_path, exist_ok=True)
os.makedirs(valid_save_path + 'image', exist_ok=True)
os.makedirs(valid_save_path + 'depth', exist_ok=True)


number_train = 47584
number_valid = 654
train_idx = 0
valid_idx = 0

for sample in tqdm(train_data, total=number_train):
    image = tf.cast(sample['image'], tf.float32)
    depth = tf.cast(sample['depth'], tf.float32)

    image = image.numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    image = image[20:460, 27:613]
    depth = depth[20:460, 27:613]

    depth = depth_inpaint(depth=depth)

    train_idx += 1
    cv2.imwrite(train_save_path + 'image/' + '_' + str(train_idx) + '.jpg', image)
    np.save(train_save_path + 'depth/' + '_' + str(train_idx) + '.npy', depth)

for sample in tqdm(valid_data, total=number_valid):
    image = tf.cast(sample['image'], tf.float32)
    depth = tf.cast(sample['depth'], tf.float32)
    
    image = image.numpy()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    image = image[20:460, 27:613]
    depth = depth[20:460, 27:613]

    depth = depth_inpaint(depth=depth)

    valid_idx += 1
    cv2.imwrite(valid_save_path + 'image/' + '_' + str(valid_idx) + '.jpg', image)
    np.save(valid_save_path + 'depth/' + '_' + str(valid_idx) + '.npy', depth)