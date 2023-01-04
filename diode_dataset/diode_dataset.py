"""diode_dataset dataset."""
import tensorflow as tf
import tensorflow_datasets as tfds
import glob
import os
import cv2
import numpy as np

# TODO(diode_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(diode_dataset): BibTeX citation
_CITATION = """
"""


class DiodeDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for diode_dataset dataset."""
  MANUAL_DOWNLOAD_INSTRUCTIONS = '/home/park/tensorflow_datasets/'
  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(diode_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3), dtype=tf.uint8),
            'depth': tfds.features.Tensor(shape=(None, None), dtype=tf.float16),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'depth'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # archive_path = '../datasets/diode_raw_datasets/'
    archive_path = dl_manager.manual_dir / 'diode_raw_datasets.zip'
    extracted_path = dl_manager.extract(archive_path)

    # TODO(cornell_grasp): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(sample_path=extracted_path/'train'),
        'validation': self._generate_examples(sample_path=extracted_path/'validation'),
    }

  def _generate_examples(self, sample_path):
    print('sample_path', sample_path)
    locations = glob.glob(str(sample_path) + '/*')
    print('locations', locations)
    idx = 0
    # 실내/ 실외 구분
    for location in locations:

        print(location)
        scenes = glob.glob(location + '/*')
        # Scene 구분
        for scene in scenes:
            scans = glob.glob(scene + '/*')
            # Scane 구분
            for scan in scans:
                data_list = glob.glob(scan + '/*.png')
                # Data 구분
                for data in data_list:
                    image = data
                    
                    # depth = '.' + data.split('.')[0] + '_depth.npy'
                    depth = data.replace('.png', '_depth.npy')
                    depth = np.load(depth).astype(np.float16)
                    # depth = tf.convert_to_tensor(depth, tf.float16)
                    # depth = tf.cast(depth, tf.float16)
                    depth = tfds.features.Encoding(depth)
                    idx += 1
                    
                    yield idx, {
                        'image': image,
                        'depth' : depth,
                    }