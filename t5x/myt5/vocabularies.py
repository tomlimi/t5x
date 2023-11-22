from typing import Any, ClassVar, Dict, Iterable, Optional, Sequence, Union
import os

from seqio import Vocabulary
import tensorflow as tf

# DECOMPOSE_MAP_PATH = os.path.join(os.path.dirname(__file__), "decompose_map.json")
# MERGE_MAP_PATH = os.path.join(os.path.dirname(__file__), "merge_map.json")


class MyteVocabulary(Vocabulary):
	"""Morphological Byte vocabulary.

	Reimplements the Byte Vocabulary method from sequio, used for ByT5 modls:
	https://arxiv.org/abs/2105.13626

	Minimal changes are applied to re-write UTF-8 to a custom standard with morphological encoding.

	"""

	DEFAULT_CONSTANT = tf.constant(-1)
	FALSE_VALUE_CONSTANT = tf.constant(0)

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

		self.rewriting = self.mock_up_tf_rewriting()
		self.rewriting_reverse = self.mock_up_tf_rewriting()

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

	@property
	def _base_vocab_size(self):
		"""Number of ids.

		Returns:
		  an integer, the vocabulary size
		"""
		return self._num_special_tokens + self._byte_size

	@classmethod
	def tf_dict(cls, keys: tf.Tensor, values: tf.Tensor) -> tf.lookup.StaticHashTable:
		return tf.lookup.StaticHashTable(
			tf.lookup.KeyValueTensorInitializer(keys, values),
			default_value=cls.DEFAULT_CONSTANT,
		)

	def mock_up_tf_rewriting(self):
		"""Mock up rewriting objects"""
		keys_input_sequences = tf.constant([bytes([i]) for i in range(self._byte_size)], dtype=tf.string)

		keys_input_subsequences = tf.constant([bytes([i]) for i in range(self._byte_size)], dtype=tf.string)
		values_output_subsequences = tf.constant([True for _ in range(self._byte_size)], dtype=tf.int32)

		output_lookup = tf.ragged.constant([[i] for i in range(self._byte_size)], dtype=tf.int32)

		input_ouput_idx_map = self.tf_dict(keys_input_sequences, tf.constant(tf.range(self._byte_size), dtype=tf.int32))
		subinput_is_terminal_map = self.tf_dict(keys_input_subsequences, values_output_subsequences)

		return input_ouput_idx_map, output_lookup, subinput_is_terminal_map

	def rewrite_bytes(self, in_bytes: tf.Tensor, input_output_idx_map, output_lookup,
	                  subinput_is_terminal_map) -> tf.Tensor:
		# rewrite bytes in tensorflow tensor
		# in_bytes_tf: tf.Tensor of shape (seq_len)
		# reverse: tf.Tensor bool, if True, apply reverse rewriting rules
		# return: tf.Tensor of shape (rewritten_seq_len)

		b_start = tf.constant(0, dtype=tf.int32)
		b_end = tf.constant(1, dtype=tf.int32)
		in_bytes_len = tf.shape(in_bytes)[0]
		output_bytes = tf.zeros([0], dtype=tf.int32)

		def inner_loop_condition(b_start, b_end, b_valid_end, cur_output_idx, keep_searching):
			not_reached_end = tf.less_equal(b_end, in_bytes_len)
			return tf.logical_and(not_reached_end, keep_searching)

		def inner_loop(b_start, b_end, b_valid_end, cur_output_idx, keep_searching):
			subinput = tf.slice(in_bytes, [b_start], [b_end - b_start])
			subinput_str = tf.strings.reduce_join(tf.gather(self._byte_strings, subinput), axis=-1)

			cur_output_idx, b_valid_end = tf.cond(
				tf.not_equal(input_output_idx_map[subinput_str], self.DEFAULT_CONSTANT),
				lambda: (input_output_idx_map[subinput_str], b_end),  # subinput is valid key, we save its index
				lambda: (cur_output_idx, b_valid_end))

			keep_searching = tf.cond(tf.equal(subinput_is_terminal_map[subinput_str], self.FALSE_VALUE_CONSTANT),
			                         lambda: tf.constant(True),
			                         lambda: tf.constant(
				                         False))  # subinput is not prefix of any longer codepoint. We break.

			return b_start, b_end + 1, b_valid_end, cur_output_idx, keep_searching

		def outer_condition(b_start, b_end, output_bytes):
			return tf.less(b_start, in_bytes_len)

		def outer_loop(b_start, b_end, output_bytes):
			cur_output_idx = self.DEFAULT_CONSTANT
			keep_searching = tf.constant(True)
			b_valid_end = b_end
			b_start, b_end, b_valid_end, cur_output_idx, keep_searching = tf.while_loop(inner_loop_condition,
			                                                                            inner_loop,
			                                                                            [b_start, b_end, b_valid_end,
			                                                                             cur_output_idx,
			                                                                             keep_searching])

			cur_output, b_end = tf.cond(tf.logical_or(tf.equal(cur_output_idx, self.DEFAULT_CONSTANT), keep_searching),
			                            lambda: (tf.slice(in_bytes, [b_start], [b_end - 1 - b_start]), b_end),
			                            # unowne codepoint, rewritting from input
			                            lambda: (output_lookup[cur_output_idx],
			                                     b_valid_end + 1))  # adding codepoint supported by maping

			output_bytes = tf.concat([output_bytes, cur_output], axis=0)
			b_start = b_end - 1

			return b_start, b_end, output_bytes

		b_start, b_end, output_bytes = tf.while_loop(outer_condition, outer_loop, [b_start, b_end, output_bytes],
		                                             shape_invariants=[tf.TensorShape([]),
		                                                               tf.TensorShape([]),
		                                                               tf.TensorShape([None])])
		return output_bytes

	def _encode(self, s):
		raise NotImplementedError

	def _decode(self, ids):
		raise NotImplementedError

	def _encode_tf(self, s):
		"""Encode a tf.Scalar string to a tf.Tensor.

		Args:
		  s: a tf.Scalar with dtype tf.string

		Returns:
		  a 1d tf.Tensor with dtype tf.int32
		"""
		ids = tf.dtypes.cast(tf.io.decode_raw(s, tf.uint8), tf.int32)

		if isinstance(ids, tf.RaggedTensor):
			ids = ids.to_tensor()
		in_shape = tf.shape(ids)
		ids = tf.reshape(ids, [-1])

		# rewrite bytes
		ids = self.rewrite_bytes(ids, *self.rewriting)

		desired_shape = tf.concat([in_shape[:-1], tf.constant([-1])], axis=0)
		ids = tf.reshape(ids, desired_shape)

		return ids + self._num_special_tokens

	def _decode_tf(self, ids):
		"""Decode in TensorFlow.

		Args:
		  ids: a n-d tf.Tensor with dtype tf.int32

		Returns:
		  a n-d tf.Tensor with dtype :string
		"""

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

		# mockup rewrite
		ids = self.rewrite_bytes(ids, *self.rewriting_reverse)

		desired_shape = tf.concat([in_shape[:-1], tf.constant([-1])], axis=0)
		ids = tf.reshape(ids, desired_shape)

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
