"""humanoid_soccer_dataset dataset."""
import json
import tensorflow_datasets as tfds
import tensorflow as tf
from pathlib import Path
import os
import time
import numpy as np
# import imageio
import cv2

# TODO(humanoid_soccer_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(humanoid_soccer_dataset): BibTeX citation
_CITATION = """
"""
CLASSES_NUMBER = 6

palette = np.array([
    [  0.],
    [  1.],
    [  2.],
    [  3.],
    [  4.],
    [  5.],
    ], dtype=np.float32)
palette = np.array([
    [180.,120.,31. ] ,
    [25.,176.,106. ] ,
    [235.,62.,156. ] ,
    [255.,255.,255.] ,
    [232.,144.,69. ] ,
    [28.,26.,227.  ] ,
    # [31.,120.,180. ] ,
    # [106.,176.,25. ] ,
    # [156.,62.,235. ] ,
    # [255.,255.,255.] ,
    # [69.,144.,232. ] ,
    # [227.,26.,28.  ] ,
    ], dtype=np.float32)


class HumanoidSoccerDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for humanoid_soccer_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(humanoid_soccer_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(240, 320, 3)),
            # 'label': tfds.features.Tensor(shape=(480, 640, CLASSES_NUMBER),dtype=tf.float32),
            'label': tfds.features.Image(shape=(240, 320, 3)),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=(None),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(humanoid_soccer_dataset): Downloads the data and defines the splits
    # archive_path = dl_manager.manual_dir / 'humanoid_Dataset.zip'
    
    # for remote chooses
    # extracted_path = dl_manager.download_and_extract('http://data.org/data.zip')
    # for local chooses
    # extracted_path = dl_manager.extract('Copy_Dataset.zip')
    extracted_path = dl_manager.extract('/home/arash/fun/Humanoid-Robots-Semantic-Segmentation/humanoid_soccer_dataset/real_dataset.zip')

    print(extracted_path)
    # TODO(humanoid_soccer_dataset): Returns the Dict[split names, Iterator[Key, Example]]

    return {
        'train': self._generate_examples(images_path = extracted_path / 'real_dataset/train/image',label_path = extracted_path / 'real_dataset/train/label'),
        'test': self._generate_examples(images_path = extracted_path / 'real_dataset/test/image',label_path = extracted_path / 'real_dataset/test/label'),
    }
  def _one_hot_encode(self,y):
    """Converts mask to a one-hot encoding specified by the semantic map."""
    img = cv2.imread(str(y), cv2.IMREAD_COLOR)
    semantic_map = []
    for color in palette:
      class_map = tf.reduce_all(tf.equal(img, color), axis=-1)
      semantic_map.append(class_map)
    semantic_map = tf.stack(semantic_map, axis=-1)
    semantic_map = tf.cast(semantic_map, tf.float32)
    semantic_map = np.array(semantic_map)
    
    
    
    magic_number = tf.reduce_sum(semantic_map)
    return semantic_map
  def _generate_examples(self, images_path, label_path):
    """Yields examples."""
    # TODO(humanoid_soccer_dataset): Yields (key, example) tuples from the dataset
    for f in images_path.glob('*.png'):
      img_name = str(f.name)
      l = Path(os.path.join(label_path,img_name))
      f = cv2.resize(cv2.imread(str(f)), (320,240))
      l = cv2.resize(cv2.imread(str(l)), (320,240), 0, 0, interpolation=cv2.INTER_NEAREST)
      # y = self._one_hot_encode(str(l))
      # count = count + 1
      # print("asdfasdfasdfa",type(y))
      yield str(f),{
        'image': f,
        'label': l,
      }
    for f in images_path.glob('*.jpg'):
      img_name = str(f.name)
      l = Path(os.path.join(label_path,img_name[:-4]+'.png'))
      f = cv2.resize(cv2.imread(str(f)), (320,240))
      l = cv2.resize(cv2.imread(str(l)), (320,240), 0, 0, interpolation=cv2.INTER_NEAREST)
      # y = self._one_hot_encode(str(l))
      # count = count + 1
      # print("asdfasdfasdfa",type(y))
      yield str(f),{
        'image': f,
        'label': l,
      }