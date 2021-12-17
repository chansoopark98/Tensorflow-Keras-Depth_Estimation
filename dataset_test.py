import tensorflow as tf
from utils.dataset_generator import DatasetGenerator

dataset = DatasetGenerator(data_dir='./datasets/', image_size=(256, 256), batch_size=1)
