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

		# for testing
		#self.decompose_forward = self.merge_forward = self.merge_reverse = self.decompose_reverse = self.mock_up_tf_rewriting()

		self.decompose_forward = self.load_tf_rewriting(os.path.join(os.path.dirname(__file__), "decompose_forward"))
		self.merge_forward = self.load_tf_rewriting(os.path.join(os.path.dirname(__file__), "merge_forward"))
		self.merge_reverse = self.load_tf_rewriting(os.path.join(os.path.dirname(__file__), "merge_reverse"))
		self.decompose_reverse = self.load_tf_rewriting(os.path.join(os.path.dirname(__file__), "decompose_reverse"))
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

	@staticmethod
	def tf_string_to_ids(string: tf.Tensor) -> tf.Tensor:
		return tf.reshape(tf.dtypes.cast(tf.io.decode_raw(string, tf.uint8), tf.int32), [-1])

	@staticmethod
	def load_tensor(load_from_dir: str, name: str, dtype) -> tf.Tensor:
		return tf.io.parse_tensor(tf.io.read_file(os.path.join(load_from_dir, name)), out_type=dtype)

	@classmethod
	def load_tensor_ragged(cls, load_from_dir: str, name: str, dtype) -> tf.RaggedTensor:
		load_to_ragged = tf.RaggedTensor.from_tensor(
			tf.expand_dims(cls.load_tensor(load_from_dir, name, dtype=tf.string), axis=1))
		return tf.map_fn(cls.tf_string_to_ids, load_to_ragged,
		                 fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=dtype), dtype=dtype)

	def mock_up_tf_rewriting(self):
		"""Mock up rewriting objects"""
		keys_input_sequences = tf.constant([bytes([i]) for i in range(self._byte_size)], dtype=tf.string)

		keys_input_subsequences = tf.constant([bytes([i]) for i in range(self._byte_size)], dtype=tf.string)
		values_output_subsequences = tf.constant([True for _ in range(self._byte_size)], dtype=tf.int32)

		output_lookup = tf.ragged.constant([[i] for i in range(self._byte_size)], dtype=tf.int32)

		input_ouput_idx_map = self.tf_dict(keys_input_sequences, tf.constant(tf.range(self._byte_size), dtype=tf.int32))
		subinput_is_terminal_map = self.tf_dict(keys_input_subsequences, values_output_subsequences)

		return input_ouput_idx_map, output_lookup, subinput_is_terminal_map

	def load_tf_rewriting(self, load_from_dir: str):
		# load tf lookup tables from path
		keys_input_sequences = self.load_tensor(load_from_dir, "keys_input_sequences", dtype=tf.string)
		output_lookup = self.load_tensor_ragged(load_from_dir, "output_lookup", dtype=tf.int32)
		keys_input_subsequences = self.load_tensor(load_from_dir, "keys_input_subsequences", dtype=tf.string)
		values_is_terminal = self.load_tensor(load_from_dir, "values_is_terminal", dtype=tf.int32)

		input_ouput_idx_map = self.tf_dict(keys_input_sequences,
		                                   tf.constant(tf.range(tf.shape(keys_input_sequences)[0]), dtype=tf.int32))
		subinput_is_terminal_map = self.tf_dict(keys_input_subsequences, values_is_terminal)
		return input_ouput_idx_map, output_lookup, subinput_is_terminal_map

	def rewrite_bytes(self, in_bytes: tf.Tensor, input_output_idx_map, output_lookup,
	                  subinput_is_terminal_map) -> tf.Tensor:

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
		@tf.function
		def encode(ids):
			ids = tf.reshape(ids, [-1])
			# 1. decompose
			ids = self.rewrite_bytes(ids, *self.decompose_forward)
			# 2. merge
			ids = self.rewrite_bytes(ids, *self.merge_forward)
			return ids

		ids = tf.dtypes.cast(tf.io.decode_raw(s, tf.uint8), tf.int32)

		expanded = False
		if ids.get_shape().ndims == 1:
			expanded = True
			ids = tf.expand_dims(ids, axis=0)

		ids = tf.map_fn(encode, ids, dtype=tf.int32,
		                fn_output_signature=tf.int32, parallel_iterations=256)

		if expanded:
			ids = tf.squeeze(ids, axis=0)

		return ids + self._num_special_tokens

	def _decode_tf(self, ids):
		"""Decode in TensorFlow.

		Args:
		  ids: a n-d tf.Tensor with dtype tf.int32

		Returns:
		  a n-d tf.Tensor with dtype :string
		"""

		@tf.function
		def decode(ids):
			ids = tf.reshape(ids, [-1])
			# 1. demerge
			ids = self.rewrite_bytes(ids, *self.merge_reverse)
			# 2. dedecompose
			ids = self.rewrite_bytes(ids, *self.decompose_reverse)
			return ids

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

		expanded = False
		if ids.get_shape().ndims == 1:
			expanded = True
			ids = tf.expand_dims(ids, axis=0)

		ids = tf.map_fn(decode, ids, dtype=tf.int32,
		                fn_output_signature=tf.int32, parallel_iterations=256)

		if expanded:
			ids = tf.squeeze(ids, axis=0)

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
