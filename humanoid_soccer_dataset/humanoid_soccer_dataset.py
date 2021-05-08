"""humanoid_soccer_dataset dataset."""
import json
import tensorflow_datasets as tfds
import tensorflow as tf
from pathlib import Path
import os
import time
import numpy as np
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

        """for remote chooses"""
        # extracted_path = dl_manager.download_and_extract('http://data.org/data.zip')
        
        """for local chooses"""
        extracted_path = dl_manager.extract('/home/arash/fun/Humanoid-Robots-Semantic-Segmentation/humanoid_soccer_dataset/real_dataset.zip')

        # TODO(humanoid_soccer_dataset): Returns the Dict[split names, Iterator[Key, Example]]
        return {
                'train': self._generate_examples(images_path = extracted_path / 'real_dataset/train/image',label_path = extracted_path / 'real_dataset/train/label'),
                'test': self._generate_examples(images_path = extracted_path / 'real_dataset/test/image',label_path = extracted_path / 'real_dataset/test/label'),
        }
        
    def _generate_examples(self, images_path, label_path):
        """Yields examples."""
        # TODO(humanoid_soccer_dataset): Yields (key, example) tuples from the dataset
        for f in images_path.glob('*.png'):
            img_name = str(f.name)
            l = Path(os.path.join(label_path,img_name))
            f = cv2.resize(cv2.imread(str(f)), (320,240))
            l = cv2.resize(cv2.imread(str(l)), (320,240), 0, 0, interpolation=cv2.INTER_NEAREST)
            yield str(f),{
                'image': f,
                'label': l,
            }
        for f in images_path.glob('*.jpg'):
            img_name = str(f.name)
            l = Path(os.path.join(label_path,img_name[:-4]+'.png'))
            f = cv2.resize(cv2.imread(str(f)), (320,240))
            l = cv2.resize(cv2.imread(str(l)), (320,240), 0, 0, interpolation=cv2.INTER_NEAREST)
            yield str(f),{
                'image': f,
                'label': l,
            }