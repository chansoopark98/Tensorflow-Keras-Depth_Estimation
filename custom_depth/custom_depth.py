"""custom_depth dataset."""
import tensorflow as tf
import tensorflow_datasets as tfds
import glob
import os
import numpy as np
import natsort

# TODO(custom_depth): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(custom_depth): BibTeX citation
_CITATION = """
"""


class CustomDepth(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for custom_depth dataset."""
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
            'image': tfds.features.Image(shape=(480, 640, 3), dtype=tf.uint8),
            'depth': tfds.features.Tensor(shape=(480, 640), dtype=tf.float16),
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
    archive_path = dl_manager.manual_dir / 'custom_dataset.zip'
    extracted_path = dl_manager.extract(archive_path)

    # TODO(cornell_grasp): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(sample_path=extracted_path),
    }

  def _generate_examples(self, sample_path):
    
    img = os.path.join(sample_path, 'image', '*.jpg')
    depth = os.path.join(sample_path, 'depth', '*.npy')
    
    img_files = glob.glob(img)
    # img_files.sort()
    img_files = natsort.natsorted(img_files,reverse=True)
    
    depth_files = glob.glob(depth)
    # mask_files.sort()
    depth_files = natsort.natsorted(depth_files,reverse=True)
    # shuffle list same orders 

    # image = image[20:460, 27:613] 440, 586
    # depth = depth[20:460, 27:613]

    
    for i in range(len(img_files)):
      yield i, {
          'image': img_files[i],
          'depth' : np.load(depth_files[i]).astype(np.float16)
      }