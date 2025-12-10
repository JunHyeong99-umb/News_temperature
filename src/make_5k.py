import random

in_path = "data/processed/train.jsonl"
out_path = "data/processed/train_5k.jsonl"
N = 5000

with open(in_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

random.seed(42)
samples = random.sample(lines, min(N, len(lines)))

with open(out_path, "w", encoding="utf-8") as w:
    w.writelines(samples)

print(f"wrote {len(samples)} lines to {out_path}")
