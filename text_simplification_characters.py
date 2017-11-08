""" 

Text Simplification - English Model

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import wsj_parsing
from tensor2tensor.data_generators.wmt import character_generator
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer

import tensorflow as tf

FLAGS = tf.flags.FLAGS

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

LOCATION_OF_DATA='/root/t2t_data/'

# Simple and Normal Wiki datasets
_TEXT_SIMPLIFICATION_TRAIN_DATASETS = [
    LOCATION_OF_DATA+'normal.training.txt',
    LOCATION_OF_DATA+'simple.training.txt'
]

_TEXT_SIMPLIFICATION_TEST_DATASETS = [
    LOCATION_OF_DATA+'normal.testing.txt',
    LOCATION_OF_DATA+'simple.testing.txt'
]

def character_generator(source_path, target_path, character_vocab, eos=None):
  """Generator for sequence-to-sequence tasks that just uses characters.

  This generator assumes the files at source_path and target_path have
  the same number of lines and yields dictionaries of "inputs" and "targets"
  where inputs are characters from the source lines converted to integers,
  and targets are characters from the target lines, also converted to integers.

  Args:
    source_path: path to the file with source sentences.
    target_path: path to the file with target sentences.
    character_vocab: a TextEncoder to encode the characters.
    eos: integer to append at the end of each sequence (default: None).

  Yields:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from characters in the file lines.
  """
  eos_list = [] if eos is None else [eos]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      while source and target:
        source_ints = character_vocab.encode(source.strip()) + eos_list
        target_ints = character_vocab.encode(target.strip()) + eos_list
        yield {"inputs": source_ints, "targets": target_ints}
        source, target = source_file.readline(), target_file.readline()


def _compile_data(tmp_dir, datasets, filename):
  """Concatenate all `datasets` and save to `filename`."""
  filename = os.path.join(tmp_dir, filename)
  with tf.gfile.GFile(filename + ".lang1", mode="w") as lang1_resfile:
    with tf.gfile.GFile(filename + ".lang2", mode="w") as lang2_resfile:
      for dataset in datasets:
        url = dataset[0]
        compressed_filename = os.path.basename(url)
        compressed_filepath = os.path.join(tmp_dir, compressed_filename)

        lang1_filename = dataset[0]
        lang2_filename = dataset[1]
        lang1_filepath = os.path.join(tmp_dir, lang1_filename)
        lang2_filepath = os.path.join(tmp_dir, lang2_filename)
        is_sgm = (lang1_filename.endswith("sgm") and
                  lang2_filename.endswith("sgm"))

  return filename


@registry.register_hparams
def text_simplification_hparams(self):
    hparams = transformer.transformer_base_single_gpu()
    hparams.batch_size = 1024
    return hparams


@registry.register_problem
class text_simplification(problem.Text2TextProblem):
  """Problem spec for text simplficiation."""
  @property
  def is_character_level(self):
    return True

  @property
  def targeted_vocab_size(self):
    return 44201

  @property
  def vocab_name(self):
    return "vocab.text_simplification.en"

  def generator(self, data_dir, tmp_dir, train):
    character_vocab = text_encoder.ByteTextEncoder()
    datasets = _TEXT_SIMPLIFICATION_TRAIN_DATASETS if train else _TEXT_SIMPLIFICATION_TEST_DATASETS
    return character_generator(datasets[0], datasets[1], character_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def num_shards(self):
    return 100

  @property
  def use_subword_tokenizer(self):
    return False
                                                                                                                    133,5         Bot