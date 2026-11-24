import os
import gc
from datasets import load_dataset

os.environ["RAYON_NUM_THREADS"] = (
    "2"  
)

wiki_stream = load_dataset(
    "wikimedia/wikipedia", "20231101.en", split="train", streaming=True
)

with open("wiki_500k.txt", "w", encoding="utf-8") as f:
  for i, example in enumerate(wiki_stream):
    if i >= 500_000:
      break
    for line in example["text"][:4000].split("\n"):
      clean_words = [w for w in line.split() if len(w) <= 35]
      if clean_words:
        f.write(" ".join(clean_words) + "\n")

del wiki_stream
gc.collect()
print("Saved wiki_500k.txt cleanly!")
