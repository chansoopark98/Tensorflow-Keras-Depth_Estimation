"""diode_dataset dataset."""
import tensorflow as tf
import tensorflow_datasets as tfds
import glob
import os
import cv2
import numpy as np
import natsort

# TODO(diode_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""
def _load_tif(path):
  with tf.io.gfile.GFile(path, "rb") as fp:
    image = tfds.core.lazy_imports.PIL_Image.open(fp)
    # rgb_img = image.convert("RGB")
  return np.array(image).astype(np.float16)
  
# TODO(diode_dataset): BibTeX citation
_CITATION = """
"""

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
            'image': tfds.features.Image(shape=(768, 1024, 3), dtype=tf.uint8),
            'depth': tfds.features.Tensor(shape=(768, 1024), dtype=tf.float16),
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
    archive_path = dl_manager.manual_dir / 'diode_indoor.zip'
    extracted_path = dl_manager.extract(archive_path)

    # TODO(cornell_grasp): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(sample_path=extracted_path/'train'),
        'validation': self._generate_examples(sample_path=extracted_path/'validation'),
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

    # img_files = np.array(img_files)
    # mask_files = np.array(mask_files)

    # indices = np.arange(img_files.shape[0])
    # np.random.shuffle(indices)

    # img_files = list(img_files[indices])
    # mask_files = list(mask_files[indices])
    
    for i in range(len(img_files)):
      yield i, {
          'image': img_files[i],
          'depth' : np.load(depth_files[i]).astype(np.float16)
      }