from typing import Any, ClassVar, Dict, Iterable, Optional, Sequence, Union
import os

from seqio import Vocabulary
import tensorflow as tf
import binascii
from tensorflow_text import WordpieceTokenizer
import os
from google.cloud import storage

BUCKET_NAME = "t5-bucket-eur"


MERGE_PRE_FILE = os.path.join(os.path.dirname(__file__), "merge_pre.txt")
MERGE_POST_FILE = os.path.join(os.path.dirname(__file__), "merge_post.txt")

DECOMPOSE_PRE_FILE = os.path.join(os.path.dirname(__file__), "decompose_pre.txt")
DECOMPOSE_POST_FILE = os.path.join(os.path.dirname(__file__), "decompose_post.txt")

DECOMPOSE_PRE_DEDUP_FILE = os.path.join(os.path.dirname(__file__), "decompose_pre_dedup.txt")
DECOMPOSE_POST_DEDUP_FILE = os.path.join(os.path.dirname(__file__), "decompose_post_dedup.txt")

MERGE_PRE_BLOB = "morphology_files/merge_pre.txt"
MERGE_POST_BLOB = "morphology_files/merge_post.txt"

DECOMPOSE_PRE_BLOB = "morphology_files/decompose_pre.txt"
DECOMPOSE_POST_BLOB = "morphology_files/decompose_post.txt"

DECOMPOSE_PRE_DEDUP_BLOB = "morphology_files/decompose_pre_dedup.txt"
DECOMPOSE_POST_DEDUP_BLOB = "morphology_files/decompose_post_dedup.txt"

PAD_ID = 0


class MyteVocabulary(Vocabulary):

  def __init__(self,  extra_ids: int = 0):
    self._byte_size = 256
    # The special tokens: 0=PAD, 1=EOS,and 2=UNK
    self._num_special_tokens = 3

    super().__init__(extra_ids=extra_ids)
    self.wordpiece_models = {}
    self.output_tensors = {}

    self.wordpiece_models['decompose'], self.output_tensors['decompose'] = self.get_wpt_and_tensor(DECOMPOSE_PRE_BLOB, DECOMPOSE_POST_BLOB,
                                                                                                   DECOMPOSE_PRE_FILE, DECOMPOSE_POST_FILE, dehexify_output=False)
    self.wordpiece_models['merge'], self.output_tensors['merge'] = self.get_wpt_and_tensor(MERGE_PRE_BLOB, MERGE_POST_BLOB,
                                                                                           MERGE_PRE_FILE, MERGE_POST_FILE,
                                                                                           dehexify_output=True)

    self.wordpiece_models['demerge'], self.output_tensors['demerge']= self.get_wpt_and_tensor(MERGE_PRE_BLOB, MERGE_POST_BLOB,
                                                                                              MERGE_POST_FILE, MERGE_PRE_FILE,
                                                                                              dehexify_output=False)
    self.wordpiece_models['dedecompose'], self.output_tensors['dedecompose'] = self.get_wpt_and_tensor(DECOMPOSE_POST_DEDUP_FILE, DECOMPOSE_PRE_DEDUP_FILE,
                                                                                                       DECOMPOSE_POST_DEDUP_FILE, DECOMPOSE_PRE_DEDUP_FILE, dehexify_output=True)



  def get_wpt_and_tensor(self, blob_pre_name: str, blob_post_name: str, file_pre: str, file_post: str, dehexify_output: bool):
    client = storage.Client()
    bucket = client.get_bucket(BUCKET_NAME)
    blob_pre = bucket.blob(blob_pre_name)
    blob_post = bucket.blob(blob_post_name)

    blob_pre.download_to_filename(file_pre)
    blob_post.download_to_filename(file_post)

    wpt = WordpieceTokenizer(file_pre,
                                 suffix_indicator = '',
                                 max_bytes_per_word = 600,
                                 token_out_type=tf.int32)

    with open(file_post, "r") as out_file:
      post_encoding_list = out_file.readlines()

    if dehexify_output:
      post_encoding_list = [binascii.unhexlify(s.strip()) for s in post_encoding_list] + [b'']
    else:
      post_encoding_list = [s.strip() for s in post_encoding_list] + [u'']

    # '' is added to account for unkonwn bytes
    post_encoding_tensor = tf.constant(post_encoding_list, dtype=tf.string)

    return wpt, post_encoding_tensor

  @property
  def _byte_strings(self):
    return tf.constant([bytes([i]) for i in range(self._byte_size)])

  @property
  def _hex_strings(self):
    return tf.constant([f"{i:02x}" for i in range(self._byte_size)])

  @property
  def _hex_strings_no_space(self):
    return tf.constant([f"{i:02x}" if i != 32 else ' ' for i in range(self._byte_size)])

  @property
  def bos_id(self) -> Optional[int]:
    return None

  @property
  def eos_id(self) -> Optional[int]:
    return 1

  @property
  def unk_id(self) -> Optional[int]:
    return 2

  @property
  def _base_vocab_size(self):
    """Number of ids.

    Returns:
      an integer, the vocabulary size
    """
    return self._num_special_tokens + self._byte_size

  def rewrite(self, seqs: tf.Tensor, phase: str) -> tf.Tensor:
    seqs = tf.strings.split(seqs, sep=' ')
    seqs = self.wordpiece_models[phase].tokenize(seqs)
    # join tokens into words
    seqs = tf.strings.reduce_join(tf.gather(self.output_tensors[phase], seqs), axis=-1)
    # join words by white space into a sentence
    return tf.strings.reduce_join(seqs, axis=-1, separator=' ')

  def _encode(self, s):
    """Encode a python string as a list of integers.

    Args:
      s: a string

    Returns:
      a list of integers (not terminated by EOS)
    """
    return self._encode_tf(s).numpy()

  def _decode(self, ids):
    """Decode a list of integers to a python string.

    Args:
      ids: a list of integers (not terminated by EOS)

    Returns:
      a string
    """
    ids = tf.constant(ids)
    str_text = self._decode_tf(ids)
    return str_text.numpy().decode("UTF-8")

  def _encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

	Args:
	  s: a tf.Scalar with dtype tf.string

	Returns:
	  a 1d tf.Tensor with dtype tf.int32
	"""

    # 0. Hexlify efficiently
    seqs = tf.dtypes.cast(tf.io.decode_raw(s, tf.uint8), tf.int32)
    seqs = tf.strings.reduce_join(tf.gather(self._hex_strings_no_space, seqs), axis=-1)

    # 1. decompose
    seqs = self.rewrite(seqs, "decompose")
    # 2. merge
    seqs = self.rewrite(seqs, "merge")
    # 3. Convert to int32
    seqs = tf.dtypes.cast(tf.io.decode_raw(seqs, tf.uint8), tf.int32)
    return seqs + self._num_special_tokens


  def _decode_tf(self, seqs):
    """Decode in TensorFlow.

  Args:
    seqs: a n-d tf.Tensor with dtype tf.int32

  Returns:
    a n-d tf.Tensor with dtype :string
  """
    lower_bound = self._num_special_tokens
    upper_bound = self._byte_size + self._num_special_tokens
    ids = tf.ragged.boolean_mask(
      data=seqs,
      mask=tf.math.logical_and(
        tf.math.greater_equal(seqs, lower_bound),
        tf.math.less(seqs, upper_bound),
      ),
    )
    ids = ids - self._num_special_tokens

    # 0. Hexlify
    seqs = tf.strings.reduce_join(tf.gather(self._hex_strings_no_space, ids), axis=-1)

    # 1. demerge
    seqs = self.rewrite(seqs, "demerge")
    # 2. dedecompose
    seqs = self.rewrite(seqs, "dedecompose")

    # 3. Return valid utf-8 strings
    return tf.strings.unicode_transcode(
      input=seqs,
      input_encoding="UTF-8",
      output_encoding="UTF-8",
      errors="ignore",
    )

  def __eq__(self, other):
    if not isinstance(other, MyteVocabulary):
      return False
    return (
            self.extra_ids == other.extra_ids
            and self.eos_id == other.eos_id
            and self.unk_id == other.unk_id
    )
