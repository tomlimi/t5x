from transformers import T5Config
import sys

size = sys.argv[1]
out_dir = sys.argv[2]

config = T5Config.from_pretrained(f"google/byt5-{size}")
config.save_pretrained(out_dir)