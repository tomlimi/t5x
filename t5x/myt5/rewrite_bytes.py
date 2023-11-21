import json
from collections import defaultdict
from typing import Union, Dict, Tuple, List
import logging
import binascii
import tensorflow as tf
import os

def hex_to_int_seq(bline: str, sep: str = ' ') -> Tuple[int]:
	return tuple(int(b, 16) for b in bline.split(sep))

class ByteRewriterTF:

	LEAF ='[LEAF]'

	BYTE_SIZE = 256

	LEAF_VAL = -100
	LEAF_VAL_CONSTANT = tf.constant(LEAF_VAL)
	DEFAULT_CONSTANT = tf.constant(-1)
	FALSE_VALUE_CONSTANT = tf.constant(0)

	def __init__(self, rewriting_rules: Union[str, Dict[str, str]]):

		if type(rewriting_rules) == str:
			with open(rewriting_rules, "r") as f:
				rewriting_rules = json.load(f)
		elif not type(rewriting_rules) == dict:
			raise ValueError(f"rewriting_rules should be either a path to json file or a dict, got {type(rewriting_rules)}")
		self.input_ouput_idx_map, self.output_lookup, self.subinput_is_terminal_map = self.prepare_tf_support(
			rewriting_rules)
		reverse_rewriting_rules = {v: k for k, v in rewriting_rules.items()}
		self.rev_input_ouput_idx_map, self.rev_output_lookup, self.rev_subinput_is_terminal_map = self.prepare_tf_support(
			reverse_rewriting_rules)


	@classmethod
	def tf_dict(cls, keys: tf.Tensor, values: tf.Tensor) -> tf.lookup.StaticHashTable:
		return tf.lookup.StaticHashTable(
			tf.lookup.KeyValueTensorInitializer(keys, values),
			default_value=cls.DEFAULT_CONSTANT,
		)


	@property
	def _byte_strings(self):
		return tf.constant([bytes([i]) for i in range(self.BYTE_SIZE)])


	def prepare_tf_support(self, rewriting_rules: Dict[str,str]) -> Tuple[tf.lookup.StaticHashTable, tf.RaggedTensor, tf.lookup.StaticHashTable]:
		# The followint method return three objects:
		# 1. input_ouput_idx_map: tf.lookup.StaticHashTable that maps input byte sequences to their output byte ids
		# 2. output_lookup: tf.RaggedTensor maps output byte ids to output byte sequence tensor
		# 3. subinput_is_terminal_map: tf.lookup.StaticHashTable that maps input byte sequences to a boolean tf constant that indicates if the input sequence is a terminal sequence (i.e. not a prefix of any other sequence)

		# start

		keys_input_sequences = [tf.constant([i], dtype=tf.int32) for i in range(self.BYTE_SIZE) if f"{i:02x}" not in rewriting_rules]
		values_output_sequences = [tf.constant([i], dtype=tf.int32) for i in range(self.BYTE_SIZE) if f"{i:02x}" not in rewriting_rules]

		# every one-byte sequence may be terminal
		terminal_subsequence_dict = {(i,): True for i in range(self.BYTE_SIZE) if f"{i:02x}" not in rewriting_rules}

		for in_sequence, out_sequence in rewriting_rules.items():
			in_sequence_tuple = hex_to_int_seq(in_sequence, ' ')
			keys_input_sequences.append(tf.constant(in_sequence_tuple, dtype=tf.int32))
			values_output_sequences.append(tf.constant(hex_to_int_seq(out_sequence, ' '), dtype=tf.int32))

			for substr_i in range(len(in_sequence_tuple)):
				terminal_subsequence_dict[in_sequence_tuple[:substr_i]] = False
			if in_sequence not in terminal_subsequence_dict:
				terminal_subsequence_dict[in_sequence_tuple] = True

		keys_input_sequences = tf.ragged.stack(keys_input_sequences)

		keys_input_sequences = tf.strings.reduce_join(tf.gather(self._byte_strings, keys_input_sequences), axis=-1)
		# save dictionary in tf lookup format
		input_ouput_idx_map = self.tf_dict(keys_input_sequences,
		                                   tf.constant(tf.range(len(keys_input_sequences), dtype=tf.int32)))

		output_lookup = tf.ragged.stack(values_output_sequences)

		keys_input_subsequences = tf.ragged.stack([tf.constant(subseq, dtype=tf.int32) for subseq in terminal_subsequence_dict.keys()])
		keys_input_subsequences = tf.strings.reduce_join(tf.gather(self._byte_strings, keys_input_subsequences), axis=-1)

		subinput_is_terminal_map = self.tf_dict(tf.stack(keys_input_subsequences, axis=0),
		                                        tf.constant(list(terminal_subsequence_dict.values()), dtype=tf.int32))

		return input_ouput_idx_map, output_lookup, subinput_is_terminal_map

	def rewrite_bytes_tf(self, in_bytes: tf.Tensor) -> tf.Tensor:
		# rewrite bytes in tensorflow tensor
		# in_bytes_tf: tf.Tensor of shape (seq_len)
		# reverse: tf.Tensor bool, if True, apply reverse rewriting rules
		# return: tf.Tensor of shape (rewritten_seq_len)

		b_start = tf.constant(0, dtype=tf.int32)
		b_end = tf.constant(1, dtype=tf.int32)
		in_bytes_len = tf.shape(in_bytes)[0]
		output_bytes = tf.zeros([0], dtype=tf.int32)

		input_ouput_idx_map, output_lookup, subinput_is_terminal_map = \
			self.input_ouput_idx_map, self.output_lookup, self.subinput_is_terminal_map

		def inner_loop_condition(b_start, b_end, b_valid_end, cur_output_idx, keep_searching):
			not_reached_end = tf.less_equal(b_end, in_bytes_len)
			return tf.logical_and(not_reached_end, keep_searching)

		def inner_loop(b_start, b_end, b_valid_end, cur_output_idx, keep_searching):
			subinput = tf.slice(in_bytes, [b_start], [b_end - b_start])
			subinput_str = tf.strings.reduce_join(tf.gather(self._byte_strings, subinput), axis=-1)

			cur_output_idx, b_valid_end = tf.cond(tf.not_equal(input_ouput_idx_map[subinput_str], self.DEFAULT_CONSTANT),
			                         lambda: (input_ouput_idx_map[subinput_str], b_end), # subinput is valid key, we save its index
			                         lambda: (cur_output_idx, b_valid_end))

			keep_searching = tf.cond(tf.equal(subinput_is_terminal_map[subinput_str], self.FALSE_VALUE_CONSTANT),
			                          lambda: tf.constant(True),
			                          lambda: tf.constant(False)) # subinput is not prefix of any longer codepoint. We break.

			return b_start, b_end + 1, b_valid_end, cur_output_idx, keep_searching

		def outer_condition(b_start, b_end, output_bytes):
			return tf.less(b_start, in_bytes_len)

		def outer_loop(b_start, b_end, output_bytes):

			cur_output_idx = self.DEFAULT_CONSTANT
			keep_searching = tf.constant(True)
			b_valid_end = b_end
			b_start, b_end, b_valid_end, cur_output_idx, keep_searching = tf.while_loop(inner_loop_condition, inner_loop,
			                                                               [b_start, b_end,b_valid_end, cur_output_idx, keep_searching])

			cur_output, b_end = tf.cond(tf.logical_or(tf.equal(cur_output_idx, self.DEFAULT_CONSTANT), keep_searching),
			                     lambda: (tf.slice(in_bytes, [b_start], [b_end - 1 - b_start]), b_end), # unowne codepoint, rewritting from input
			                     lambda: (output_lookup[cur_output_idx], b_valid_end + 1)) # adding codepoint supported by maping

			output_bytes = tf.concat([output_bytes, cur_output], axis=0)
			b_start = b_end - 1

			return b_start, b_end, output_bytes

		b_start, b_end, output_bytes = tf.while_loop(outer_condition, outer_loop, [b_start, b_end, output_bytes],
		                                             shape_invariants=[tf.TensorShape([]),
		                                                               tf.TensorShape([]),
		                                                               tf.TensorShape([None])])
		return output_bytes

	def rewrite_bytes_tf_reverse(self, in_bytes: tf.Tensor) -> tf.Tensor:
		# rewrite bytes in tensorflow tensor
		# in_bytes_tf: tf.Tensor of shape (seq_len)
		# reverse: tf.Tensor bool, if True, apply reverse rewriting rules
		# return: tf.Tensor of shape (rewritten_seq_len)

		b_start = tf.constant(0, dtype=tf.int32)
		b_end = tf.constant(1, dtype=tf.int32)
		in_bytes_len = tf.shape(in_bytes)[0]
		output_bytes = tf.zeros([0], dtype=tf.int32)

		input_ouput_idx_map, output_lookup, subinput_is_terminal_map = \
			self.rev_input_ouput_idx_map, self.rev_output_lookup, self.rev_subinput_is_terminal_map

		def inner_loop_condition(b_start, b_end, b_valid_end, cur_output_idx, keep_searching):
			not_reached_end = tf.less_equal(b_end, in_bytes_len)
			return tf.logical_and(not_reached_end, keep_searching)

		def inner_loop(b_start, b_end, b_valid_end, cur_output_idx, keep_searching):
			subinput = tf.slice(in_bytes, [b_start], [b_end - b_start])
			subinput_str = tf.strings.reduce_join(tf.gather(self._byte_strings, subinput), axis=-1)

			cur_output_idx, b_valid_end = tf.cond(tf.not_equal(input_ouput_idx_map[subinput_str], self.DEFAULT_CONSTANT),
			                         lambda: (input_ouput_idx_map[subinput_str], b_end), # subinput is valid key, we save its index
			                         lambda: (cur_output_idx, b_valid_end))

			keep_searching = tf.cond(tf.equal(subinput_is_terminal_map[subinput_str], self.FALSE_VALUE_CONSTANT),
			                          lambda: tf.constant(True),
			                          lambda: tf.constant(False)) # subinput is not prefix of any longer codepoint. We break.

			return b_start, b_end + 1, b_valid_end, cur_output_idx, keep_searching

		def outer_condition(b_start, b_end, output_bytes):
			return tf.less(b_start, in_bytes_len)

		def outer_loop(b_start, b_end, output_bytes):

			cur_output_idx = self.DEFAULT_CONSTANT
			keep_searching = tf.constant(True)
			b_valid_end = b_end
			b_start, b_end, b_valid_end, cur_output_idx, keep_searching = tf.while_loop(inner_loop_condition, inner_loop,
			                                                               [b_start, b_end,b_valid_end, cur_output_idx, keep_searching])

			cur_output, b_end = tf.cond(tf.logical_or(tf.equal(cur_output_idx, self.DEFAULT_CONSTANT), keep_searching),
			                     lambda: (tf.slice(in_bytes, [b_start], [b_end - 1 - b_start]), b_end), # unowne codepoint, rewritting from input
			                     lambda: (output_lookup[cur_output_idx], b_valid_end + 1)) # adding codepoint supported by maping

			output_bytes = tf.concat([output_bytes, cur_output], axis=0)
			b_start = b_end - 1

			return b_start, b_end, output_bytes

		b_start, b_end, output_bytes = tf.while_loop(outer_condition, outer_loop, [b_start, b_end, output_bytes],
		                                             shape_invariants=[tf.TensorShape([]),
		                                                               tf.TensorShape([]),
		                                                               tf.TensorShape([None])])
		return output_bytes
