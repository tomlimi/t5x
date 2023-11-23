import json
from collections import defaultdict
from typing import Union, Dict, Tuple, List
import logging
import binascii
import tensorflow as tf
import os

def hex_to_int_seq(bline: str, sep: str = ' ') -> Tuple[int]:
	return tuple(int(b, 16) for b in bline.split(sep))

DECOMPOSE_MAP_PATH = os.path.join(os.path.dirname(__file__), "decompose_map.json")
MERGE_MAP_PATH = os.path.join(os.path.dirname(__file__), "merge_map.json")

class ByteRewriterTF:

	LEAF ='[LEAF]'

	BYTE_SIZE = 256

	LEAF_VAL = -100
	LEAF_VAL_CONSTANT = tf.constant(LEAF_VAL)
	DEFAULT_CONSTANT = tf.constant(-1)
	FALSE_VALUE_CONSTANT = tf.constant(0)

	def __init__(self, rewriting_rules: Union[str, Dict[str, str]], save_tensorflow: str = None, from_tensorflow: bool = False):

		if type(rewriting_rules) == str:
			with open(rewriting_rules, "r") as f:
				rewriting_rules = json.load(f)
		elif not type(rewriting_rules) == dict:
			raise ValueError(f"rewriting_rules should be either a path to json file or a dict, got {type(rewriting_rules)}")

		self.input_ouput_idx_map, self.output_lookup, self.subinput_is_terminal_map = self.prepare_tf_support(
			rewriting_rules, save_to_dir=save_tensorflow + "_forward" if save_tensorflow is not None else None)
		reverse_rewriting_rules = {v: k for k, v in rewriting_rules.items()}
		self.rev_input_ouput_idx_map, self.rev_output_lookup, self.rev_subinput_is_terminal_map = self.prepare_tf_support(
			reverse_rewriting_rules, save_to_dir=save_tensorflow + "_reverse" if save_tensorflow is not None else None)


	@classmethod
	def tf_dict(cls, keys: tf.Tensor, values: tf.Tensor) -> tf.lookup.StaticHashTable:
		return tf.lookup.StaticHashTable(
			tf.lookup.KeyValueTensorInitializer(keys, values),
			default_value=cls.DEFAULT_CONSTANT,
		)


	@property
	def _byte_strings(self):
		return tf.constant([bytes([i]) for i in range(self.BYTE_SIZE)])

	def tf_ids_to_string(self, ids: tf.Tensor) -> tf.Tensor:
		return tf.strings.reduce_join(tf.gather(self._byte_strings, ids), axis=-1)

	@staticmethod
	def tf_string_to_ids(string: tf.Tensor) -> tf.Tensor:
		return tf.reshape(tf.dtypes.cast(tf.io.decode_raw(string, tf.uint8), tf.int32), [-1])


	@staticmethod
	def save_tensor(save_to_dir: str, name: str, tensor: tf.Tensor):
		tf.io.write_file(os.path.join(save_to_dir,name), tf.io.serialize_tensor(tensor))

	def save_tensor_ragged(self, save_to_dir: str, name: str, r_tensor: tf.RaggedTensor):
		self.save_tensor(save_to_dir, name, tf.map_fn(self.tf_ids_to_string, r_tensor, dtype=tf.string))

	@staticmethod
	def load_tensor(load_from_dir: str, name: str, dtype) -> tf.Tensor:
		return tf.io.parse_tensor(tf.io.read_file(os.path.join(load_from_dir,name)), out_type=dtype)

	@classmethod
	def load_tensor_ragged(cls, load_from_dir: str, name: str, dtype) -> tf.RaggedTensor:
		load_to_ragged = tf.RaggedTensor.from_tensor(tf.expand_dims(cls.load_tensor(load_from_dir, name, dtype=dtype), axis=1))
		return tf.map_fn(cls.tf_string_to_ids,load_to_ragged,
		                 fn_output_signature=tf.RaggedTensorSpec(shape=[None], dtype=tf.int32), dtype=tf.int32)

	def prepare_tf_support(self, rewriting_rules: Dict[str,str], save_to_dir: str = None) -> Tuple[tf.lookup.StaticHashTable, tf.RaggedTensor, tf.lookup.StaticHashTable]:
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

		values_is_terminal = tf.constant(list(terminal_subsequence_dict.values()), dtype=tf.int32)

		subinput_is_terminal_map = self.tf_dict(tf.stack(keys_input_subsequences, axis=0),
		                                        values_is_terminal)

		if save_to_dir is not None:
			print("Saving tf lookup tables to", save_to_dir)
			self.save_tensor(save_to_dir, "keys_input_sequences", keys_input_sequences)
			self.save_tensor_ragged(save_to_dir, "output_lookup", output_lookup)
			self.save_tensor(save_to_dir, "keys_input_subsequences", keys_input_subsequences)
			self.save_tensor(save_to_dir, "values_is_terminal", values_is_terminal)

		return input_ouput_idx_map, output_lookup, subinput_is_terminal_map

	def load_tf_support(self, load_from_dir: str):
		# load tf lookup tables from path
		keys_input_sequences = self.load_tensor(load_from_dir, "keys_input_sequences", dtype=tf.string)
		output_lookup = self.load_tensor_ragged(load_from_dir, "output_lookup", dtype=tf.int32)
		keys_input_subsequences = self.load_tensor(load_from_dir, "keys_input_subsequences", dtype=tf.string)
		values_is_terminal = self.load_tensor(load_from_dir, "values_is_terminal", dtype=tf.int32)

		input_ouput_idx_map = self.tf_dict(keys_input_sequences,
		                                   tf.constant(tf.range(len(keys_input_sequences), dtype=tf.int32)))
		subinput_is_terminal_map = self.tf_dict(keys_input_subsequences,
		                                        values_is_terminal)
		return input_ouput_idx_map, output_lookup, subinput_is_terminal_map


if __name__ == "__main__":

	# load rewriting rules and write them to tf lookup tables
	decompose_rewriter = ByteRewriterTF(DECOMPOSE_MAP_PATH, save_tensorflow=os.path.join(os.path.dirname(__file__),"decompose"))

	merge_rewriter = ByteRewriterTF(MERGE_MAP_PATH, save_tensorflow=os.path.join(os.path.dirname(__file__),"merge"))