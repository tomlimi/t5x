import json
from collections import defaultdict
from typing import Union, Dict, Tuple, List
import logging
import binascii
import tensorflow as tf

def hex_to_int_seq(bline: str, sep: str = ' ') -> Tuple[int]:
	return tuple(int(b, 16) for b in bline.split(sep))

class ByteRewriter:

	LEAF ='[LEAF]'

	def __init__(self, rewriting_rules: Union[str, Dict[str, str]]):

		if type(rewriting_rules) == str:
			with open(rewriting_rules, "r") as f:
				rewriting_rules = json.load(f)
		elif not type(rewriting_rules) == dict:
			raise ValueError(f"rewriting_rules should be either a path to json file or a dict, got {type(rewriting_rules)}")

		self.hash_tree = self.construct_hash_tree(rewriting_rules)
		reverse_rewriting_rules = {v:k for k,v in rewriting_rules.items()}
		self.reverse_hash_tree = self.construct_hash_tree(reverse_rewriting_rules)

	def add_leaf(self,hash_tree, byte_in_sequence, byte_out_sequence):

		byte_in_list = hex_to_int_seq(byte_in_sequence, ' ')
		byte_out_list = hex_to_int_seq(byte_out_sequence, ' ')

		tree_pointer = hash_tree
		for b in byte_in_list:
			if b not in tree_pointer:
				tree_pointer[b] = {}
			tree_pointer = tree_pointer[b]

		tree_pointer[self.LEAF] = byte_out_list

	def construct_hash_tree(self, rewriting_rules):

		hash_tree = defaultdict(dict)
		for b in range(256):
			hash_tree[b][self.LEAF] = [b]

		for in_sequence, out_sequence in rewriting_rules.items():
			self.add_leaf(hash_tree, in_sequence, out_sequence)

		return hash_tree

	def search_hash_tree(self, byte_sequence):

		tree_pointer = self.hash_tree
		for b in byte_sequence:
			if b in tree_pointer:
				tree_pointer = tree_pointer[b]
			else:
				return None

		return tree_pointer[self.LEAF]

	def rewrite_bytes(self, in_bytes: List[int], reverse=False) -> List[int]:

		out_bytes = []
		b_start = 0
		b_end = 0

		while b_start < len(in_bytes):
			tree_pointer = self.hash_tree if not reverse else self.reverse_hash_tree
			for j in range(b_start, len(in_bytes)):
				b = in_bytes[j]
				if b in tree_pointer:
					tree_pointer = tree_pointer[b]
				elif j == b_start:
					logging.warning(f"Unrecognized byte {b} in {in_bytes}, Skipping!")
					cur_leaf = [b]
					b_end = j
					break
				else:
					break
				if self.LEAF in tree_pointer:
					cur_leaf = tree_pointer[self.LEAF]
					b_end = j
			out_bytes.extend(cur_leaf)
			b_start = b_end + 1

		return out_bytes

	def rewrite_bytes_tf(self, in_bytes: tf.Tensor, reverse=False) -> List[int]:
		# We want to return the tensor with the same dimensionality as before
		if isinstance(in_bytes, tf.RaggedTensor):
			in_bytes = in_bytes.to_tensor()
		in_shape = tf.shape(in_bytes)
		in_bytes = tf.reshape(in_bytes, [-1])
		in_bytes = in_bytes.numpy()

		out_bytes = []
		b_start = 0
		b_end = 0

		in_bytes_len = len(in_bytes)
		while b_start < in_bytes_len:
			tree_pointer = self.hash_tree if not reverse else self.reverse_hash_tree
			for j in range(b_start, in_bytes_len):
				b = in_bytes[j]
				if b in tree_pointer:
					tree_pointer = tree_pointer[b]
				elif j == b_start:
					logging.warning(f"Unrecognized byte {b} in {in_bytes}, Skipping!")
					cur_leaf = [b]
					b_end = j
					break
				else:
					break
				if self.LEAF in tree_pointer:
					cur_leaf = tree_pointer[self.LEAF]
					b_end = j
			out_bytes.extend(cur_leaf)
			b_start = b_end + 1

		out_bytes_len = len(out_bytes)
		out_bytes = tf.constant(out_bytes, dtype=tf.int32)
		desired_shape = tf.concat([in_shape[:-1], tf.constant([out_bytes_len], dtype=tf.int32)], axis=0)
		out_bytes = tf.reshape(out_bytes, desired_shape)

		return out_bytes