import json
from collections import defaultdict
from typing import Union
import logging
import binascii
import tensorflow as tf

def str_to_hex(line: str, sep: str = ' ') -> str:
	return bytes_to_hex(bytes(line, 'utf-8'), sep)

def bytes_to_hex(bline: bytes, sep: str = ' ') -> str:
	return str(binascii.hexlify(bline, sep), "utf-8")

def hex_to_bytes(bline: str, sep: str = ' ') -> bytes:
	return binascii.unhexlify(bline.replace(sep, ''))

def hex_to_str(bline: str, sep: str = ' ') -> str:
	return str(binascii.unhexlify(bline.replace(sep, '')), "utf-8")

class ByteRewriter:

	LEAF ='[LEAF]'

	def __init__(self, rewriting_rules: Union[str, dict[str, str]]):

		if type(rewriting_rules) == str:
			with open(rewriting_rules, "r") as f:
				rewriting_rules = json.load(f)
		elif not type(rewriting_rules) == dict:
			raise ValueError(f"rewriting_rules should be either a path to json file or a dict, got {type(rewriting_rules)}")

		self.hash_tree = self.construct_hash_tree(rewriting_rules)
		revese_revrewriting_rules = {v:k for k,v in rewriting_rules.items()}
		self.reverse_hash_tree = self.construct_hash_tree(revese_revrewriting_rules)

	def add_leaf(self,hash_tree, byte_in_sequence, byte_out_sequence):

		byte_in_list = byte_in_sequence.split(' ')
		byte_out_list = byte_out_sequence.split(' ')

		tree_pointer = hash_tree
		for b in byte_in_list:
			if b not in tree_pointer:
				tree_pointer[b] = {}
			tree_pointer = tree_pointer[b]

		tree_pointer[self.LEAF] = byte_out_list

	def construct_hash_tree(self, rewritting_rules):

		hash_tree = defaultdict(dict)
		for b in (f"{x:02x}" for x in range(256)):
			hash_tree[b][self.LEAF] = [b]

		for in_seequence, out_sequence in rewritting_rules.items():
			self.add_leaf(hash_tree, in_seequence, out_sequence)

		return hash_tree

	def search_hash_tree(self, byte_sequence):

		tree_pointer = self.hash_tree
		for b in byte_sequence:
			if b in tree_pointer:
				tree_pointer = tree_pointer[b]
			else:
				return None

		return tree_pointer[self.LEAF]

	def rewrite_bytes(self, in_bytes, reverse=False):

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

	def rewrite_bytes_tf(self, in_bytes_tf, reverse=False):
		# rewrite bytes in tensorflow tensor
		# in_bytes_tf: tf.Tensor of shape (batch_size, seq_len)
		# reverse: bool, if True, apply reverse rewriting rules
		# return: tf.Tensor of shape (batch_size, rewritten_seq_len)

		out_bytes_tf = []

		for i in range(in_bytes_tf.shape[0]):
			out_bytes_tf.append(self.rewrite_bytes(in_bytes_tf[i], reverse=reverse))

		return tf.ragged.constant(out_bytes_tf)