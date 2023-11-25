import json
import os


merge_dict_file = os.path.join(os.path.dirname(__file__), "merge_map.json")
decompose_dict_file = os.path.join(os.path.dirname(__file__), "decompose_map.json")

merge_pre_text_file = os.path.join(os.path.dirname(__file__), "merge_pre.txt")
merge_post_text_file = os.path.join(os.path.dirname(__file__), "merge_post.txt")

decompose_pre_text_file = os.path.join(os.path.dirname(__file__), "decompose_pre.txt")
decompose_post_text_file = os.path.join(os.path.dirname(__file__), "decompose_post.txt")

decompose_pre_dedup_text_file = os.path.join(os.path.dirname(__file__), "decompose_pre_dedup.txt")
decompose_post_dedup_text_file = os.path.join(os.path.dirname(__file__), "decompose_post_dedup.txt")

ONE_BYTE_LIST = [f"{i:02x}" for i in range(256)]

def load_json(file_path):
	with open(file_path, "r") as f:
		return json.load(f)

def write_text_file(file_path, byte_words):

	with open(file_path, "w") as f:
		byte_words = [bw.replace(' ','').strip() for bw in byte_words]
		f.write("\n".join(byte_words))


merge_dict = load_json(merge_dict_file)
decompose_dict = load_json(decompose_dict_file)

for one_byte_word in ONE_BYTE_LIST:
	if one_byte_word not in merge_dict:
		merge_dict[one_byte_word] = one_byte_word

	if one_byte_word not in decompose_dict:
		decompose_dict[one_byte_word] = one_byte_word

write_text_file(merge_pre_text_file, merge_dict.keys())
write_text_file(merge_post_text_file, merge_dict.values())

write_text_file(decompose_pre_text_file, decompose_dict.keys())
write_text_file(decompose_post_text_file, decompose_dict.values())

# deduplicate values in decompose_dict
decompose_dict_dedup = {}
for key, value in decompose_dict.items():
	if value not in decompose_dict_dedup.values():
		decompose_dict_dedup[key] = value

write_text_file(decompose_pre_dedup_text_file, decompose_dict_dedup.keys())
write_text_file(decompose_post_dedup_text_file, decompose_dict_dedup.values())





