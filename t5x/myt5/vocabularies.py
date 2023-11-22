from typing import Any, ClassVar, Dict, Iterable, Optional, Sequence, Union
import os

from seqio import Vocabulary
import tensorflow as tf
from t5x.myt5.rewrite_bytes import ByteRewriterTF

DECOMPOSE_MAP_PATH = os.path.join(os.path.dirname(__file__), "decompose_map.json")
MERGE_MAP_PATH = os.path.join(os.path.dirname(__file__), "merge_map.json")


# resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
# tf.config.experimental_connect_to_cluster(resolver)
# tf.tpu.experimental.initialize_tpu_system(resolver)
# strategy = tf.distribute.experimental.TPUStrategy(resolver)


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
		self.decompose_rewriter = ByteRewriterTF(DECOMPOSE_MAP_PATH)
		self.merge_rewriter = ByteRewriterTF(MERGE_MAP_PATH)

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
		raise NotImplementedError
		# ids_raw = list(s.encode("utf-8"))
		# ids_decomposed = self.decompose_rewriter.rewrite_bytes(ids_raw)
		# ids_merged = self.merge_rewriter.rewrite_bytes(ids_decomposed)
		# return ids_merged

	def _convert_ids_to_strings(self, ids):
		"""Convert ids to a python string based on UTF-8 encoding.

		Args:
		  ids: a list of integers

		Returns:
		  s: a string
		"""
		raise NotImplementedError
		# demerged_ids = self.merge_rewriter.rewrite_bytes(ids, reverse=True)
		# dedecomposed_ids = self.decompose_rewriter.rewrite_bytes(demerged_ids, reverse=True)
		# return bytes(dedecomposed_ids).decode("utf-8", errors="ignore")

	def _filter_non_string_ids(self, ids):
		"""Filter special token ids and extra ids if there are any.

		Args:
		  ids: a list of integers

		Returns:
		  ids: a list of integers
		"""
		raise NotImplementedError
		# lower_bound = self._num_special_tokens
		# upper_bound = self._byte_size + self._num_special_tokens
		# return [id for id in ids if lower_bound <= id < upper_bound]

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
		raise NotImplementedError
		# s = s.decode() if isinstance(s, bytes) else s
		# ids = self._convert_strings_to_ids(s)
		# return [i + self._num_special_tokens for i in ids]

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
		raise NotImplementedError
		# ids = [int(i) for i in ids]
		# ids = self._filter_non_string_ids(ids)
		# ids = [i - self._num_special_tokens for i in ids]
		# return self._convert_ids_to_strings(ids)

	def _encode_tf(self, s):
		"""Encode a tf.Scalar string to a tf.Tensor.

		Args:
		  s: a tf.Scalar with dtype tf.string

		Returns:
		  a 1d tf.Tensor with dtype tf.int32
		"""

		# @tf.function
		# def encode_byte_rewrite(in_bytes: tf.Tensor):
		# 	if isinstance(in_bytes, tf.RaggedTensor):
		# 		in_bytes = in_bytes.to_tensor()
		# 	in_shape = tf.shape(in_bytes)
		# 	bytes = tf.reshape(in_bytes, [-1])
		#
		# 	# 1. decomposing
		# 	bytes = self.decompose_rewriter.rewrite_bytes_tf(bytes)
		# 	# 2. merging
		# 	bytes = self.merge_rewriter.rewrite_bytes_tf(bytes)
		#
		# 	desired_shape = tf.concat([in_shape[:-1], tf.constant([-1])], axis=0)
		# 	return tf.reshape(bytes, desired_shape)
		#
		ids = tf.dtypes.cast(tf.io.decode_raw(s, tf.uint8), tf.int32)

		# expanded = False
		# if ids.get_shape().ndims == 1:
		# 	expanded = True
		# 	ids = tf.expand_dims(ids, axis=0)

		if isinstance(ids, tf.RaggedTensor):
			ids = ids.to_tensor()
		in_shape = tf.shape(ids)
		ids = tf.reshape(ids, [-1])

		# 1. decomposing
		ids = self.decompose_rewriter.rewrite_bytes_tf(ids)
		# 2. merging
		ids = self.merge_rewriter.rewrite_bytes_tf(ids)

		desired_shape = tf.concat([in_shape[:-1], tf.constant([-1])], axis=0)
		ids = tf.reshape(ids, desired_shape)

		# ids = tf.map_fn(encode_byte_rewrite, ids, dtype=tf.int32, fn_output_signature=tf.int32, parallel_iterations=256)
		#
		# if expanded:
		# 	ids = tf.squeeze(ids, axis=0)
		return ids + self._num_special_tokens

	def _decode_tf(self, ids):
		"""Decode in TensorFlow.

		Args:
		  ids: a n-d tf.Tensor with dtype tf.int32

		Returns:
		  a n-d tf.Tensor with dtype :string
		"""
		# @tf.function
		# def decode_byte_rewrite(in_bytes: tf.Tensor):
		# 	if isinstance(in_bytes, tf.RaggedTensor):
		# 		in_bytes = in_bytes.to_tensor()
		# 	in_shape = tf.shape(in_bytes)
		# 	bytes = tf.reshape(in_bytes, [-1])
		#
		# 	# 1. demerging
		# 	bytes = self.merge_rewriter.rewrite_bytes_tf_reverse(bytes)
		# 	# 2. dedecomposing
		# 	bytes = self.decompose_rewriter.rewrite_bytes_tf_reverse(bytes)
		#
		# 	desired_shape = tf.concat([in_shape[:-1], tf.constant([-1])], axis=0)
		# 	return tf.reshape(bytes, desired_shape)


		lower_bound = self._num_special_tokens
		upper_bound = self._byte_size + self._num_special_tokens
		ids = tf.ragged.boolean_mask(
			data=ids,
			mask=tf.math.logical_and(
				tf.math.greater_equal(ids, lower_bound),
				tf.math.less(ids, upper_bound),
			),
		)
		ids = ids - self._num_special_tokens

		if isinstance(ids, tf.RaggedTensor):
			ids = ids.to_tensor()
		in_shape = tf.shape(ids)
		ids = tf.reshape(ids, [-1])

		# 1. demerging
		ids = self.merge_rewriter.rewrite_bytes_tf_reverse(ids)
		# 2. dedecomposing
		ids = self.decompose_rewriter.rewrite_bytes_tf_reverse(ids)

		desired_shape = tf.concat([in_shape[:-1], tf.constant([-1])], axis=0)
		ids = tf.reshape(ids, desired_shape)

		# expanded = False
		# if ids.get_shape().ndims == 1:
		# 	expanded = True
		# 	ids = tf.expand_dims(ids, axis=0)
		#
		# ids = tf.map_fn(decode_byte_rewrite, ids, dtype=tf.int32,
		#                            fn_output_signature=tf.int32, parallel_iterations=256)
		#
		# if expanded:
		# 	ids = tf.squeeze(ids, axis=0)

		string = tf.strings.reduce_join(tf.gather(self._byte_strings, ids), axis=-1)
		return tf.strings.unicode_transcode(
			input=string,
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
