from typing import Any, ClassVar, Dict, Iterable, Optional, Sequence, Union
import os

from seqio import Vocabulary
import tensorflow as tf
from t5x.myt5.rewrite_bytes import ByteRewriter, hex_to_bytes, str_to_hex, bytes_to_hex, hex_to_str

DECOMPOSE_MAP_PATH = os.path.join(os.path.dirname(__file__), "decompose_map.json")
MERGE_MAP_PATH = os.path.join(os.path.dirname(__file__), "decompose_map.json")

class MyteVocabulary(Vocabulary):
	"""Morphological Byte vocabulary.

	Reimplements the Byte Vocabulary method from sequio, used for ByT5 modls:
	https://arxiv.org/abs/2105.13626

	Minimal changes are applied to re-write UTF-8 to a custom standard with morphological encoding.

	"""

	def __init__(self, extra_ids: int = 0):
		"""Create a MyteVocabulary.

		Optionally, specify a number of extra ids to add to the end of the
		vocabulary for use as sentinels.

		Args:
		  extra_ids: an optional integer
		"""
		self._byte_size = 256
		# The special tokens: 0=PAD, 1=EOS,and 2=UNK
		self._num_special_tokens = 3
		self.decompose_rewriter = ByteRewriter(DECOMPOSE_MAP_PATH)
		self.merge_rewriter = ByteRewriter(MERGE_MAP_PATH)

		super().__init__(extra_ids=extra_ids)

	@property
	def _byte_strings(self):
		return tf.constant([bytes([i]) for i in range(self._byte_size)])

	@property
	def bos_id(self) -> Optional[int]:
		return None

	@property
	def eos_id(self) -> Optional[int]:
		return 1

	@property
	def unk_id(self) -> Optional[int]:
		return 2

	def _convert_strings_to_ids(self, s):
		"""Convert a python string to integers based on UTF-8 encoding.

		Args:
		  s: a string

		Returns:
		  ids: a list of integers
		"""

		hex_sequence = str_to_hex(s).split(' ')
		decomposed_hex_sequence = self.decompose_rewriter.rewrite_bytes(hex_sequence)
		merged_hex_sequence = self.merge_rewriter.rewrite_bytes(decomposed_hex_sequence)
		return list(hex_to_bytes(' '.join(merged_hex_sequence)))

	def _convert_ids_to_strings(self, ids):
		"""Convert ids to a python string based on UTF-8 encoding.

		Args:
		  ids: a list of integers

		Returns:
		  s: a string
		"""
		hex_sequence = bytes_to_hex(bytes(ids))
		demerged_hex_sequence = self.merge_rewriter.rewrite_bytes(hex_sequence.split(' '), reverse=True)
		dedecomposed_hex_sequence = self.decompose_rewriter.rewrite_bytes(demerged_hex_sequence, reverse=True)
		return hex_to_str(' '.join(dedecomposed_hex_sequence))

	def _filter_non_string_ids(self, ids):
		"""Filter special token ids and extra ids if there are any.

		Args:
		  ids: a list of integers

		Returns:
		  ids: a list of integers
		"""
		lower_bound = self._num_special_tokens
		upper_bound = self._byte_size + self._num_special_tokens
		return [id for id in ids if lower_bound <= id < upper_bound]

	@property
	def _base_vocab_size(self):
		"""Number of ids.

		Returns:
		  an integer, the vocabulary size
		"""
		return self._num_special_tokens + self._byte_size

	def _encode(self, s):
		"""Encode a python string as a list of integers.

		To keep the first few ids for special tokens, increase ids by the number
		of special tokens.

		Args:
		  s: a string

		Returns:
		  a list of integers (not terminated by EOS)
		"""
		s = s.decode() if isinstance(s, bytes) else s
		ids = self._convert_strings_to_ids(s)
		return [i + self._num_special_tokens for i in ids]

	def _decode(self, ids):
		"""Decode a list of integers to a python string.

		The special tokens of PAD, EOS, and UNK will not be represented in the
		output string. This is different from the SentencePieceVocabulary, where
		UNK will show up as a '?' character.

		Args:
		  ids: a list of integers (not terminated by EOS)

		Returns:
		  a string
		"""
		ids = [int(i) for i in ids]
		ids = self._filter_non_string_ids(ids)
		ids = [i - self._num_special_tokens for i in ids]
		return self._convert_ids_to_strings(ids)

	def _encode_tf(self, s):
		"""Encode a tf.Scalar string to a tf.Tensor.

		Args:
		  s: a tf.Scalar with dtype tf.string

		Returns:
		  a 1d tf.Tensor with dtype tf.int32
		"""

		# cast to string

		input_string = s.numpy()
		output_bytes = self._encode(input_string)
		return tf.constant(output_bytes, dtype=tf.int32)

		# raise NotImplementedError
		# return (
		# 		tf.dtypes.cast(tf.io.decode_raw(s, tf.uint8), tf.int32)
		# 		+ self._num_special_tokens
		# )

	def _decode_tf(self, ids):
		"""Decode in TensorFlow.

		Args:
		  ids: a n-d tf.Tensor with dtype tf.int32

		Returns:
		  a n-d tf.Tensor with dtype tf.string
		"""
		input_string = ids.numpy()
		output_bytes = self._decode(input_string)
		return tf.constant(output_bytes, dtype=tf.string)

		# TODO decode with tensorflow
		# raise NotImplementedError

		# lower_bound = self._num_special_tokens
		# upper_bound = self._byte_size + self._num_special_tokens
		# ids = tf.ragged.boolean_mask(
		# 	data=ids,
		# 	mask=tf.math.logical_and(
		# 		tf.math.greater_equal(ids, lower_bound),
		# 		tf.math.less(ids, upper_bound),
		# 	),
		# )
		# ids = ids - self._num_special_tokens
		# string = tf.strings.reduce_join(tf.gather(self._byte_strings, ids), axis=-1)
		#
		# # Drop invalid byte sequences.
		# return tf.strings.unicode_transcode(
		# 	input=string,
		# 	input_encoding="UTF-8",
		# 	output_encoding="UTF-8",
		# 	errors="ignore",
		# )

	def __eq__(self, other):
		if not isinstance(other, MyteVocabulary):
			return False
		return (
				self.extra_ids == other.extra_ids
				and self.eos_id == other.eos_id
				and self.unk_id == other.unk_id
		)
