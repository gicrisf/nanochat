"""
Data preparation task for nanochat. Run this once before training.

- Downloads parquet shards via StorageManager (cached per URL in ~/.clearml/cache,
  so re-running on the same agent machine skips re-downloading)
- Trains the tokenizer
- Stores the tokenizer as a versioned ClearML Dataset reusable by any training task

Submit with:
    clearml-task --project nanochat --name data-prep --queue <queue> \
        --repo <repo-url> --script scripts/clearml_data.py \
        --docker nvcr.io/nvidia/pytorch:24.12-py3 --packages clearml
"""
import os
import subprocess
import argparse

from clearml import Task, Dataset, StorageManager
from nanochat.common import get_base_dir
from nanochat.dataset import BASE_URL, index_to_filename

DATASET_NAME = "nanochat-tokenizer"

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=str, default="nanochat")
parser.add_argument("--num-shards", type=int, default=370)
args = parser.parse_args()

task = Task.init(project_name=args.project, task_name="data-prep")
task.connect(vars(args), name="config")

base_dir = get_base_dir()
data_dir = os.path.join(base_dir, "base_data")
tokenizer_dir = os.path.join(base_dir, "tokenizer")
os.makedirs(data_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Shards: download via StorageManager so each URL is cached in ~/.clearml/cache.
# Subsequent tasks on the same agent reuse that cache via symlinks.
print(f"Fetching {args.num_shards} shards...")
for i in range(args.num_shards):
    filename = index_to_filename(i)
    link_path = os.path.join(data_dir, filename)
    if os.path.exists(link_path):
        continue
    cached_path = StorageManager.get_local_copy(f"{BASE_URL}/{filename}")
    os.symlink(cached_path, link_path)
print("Shards ready.")

# -----------------------------------------------------------------------------
# Tokenizer: train if not already present, then upload as a ClearML Dataset.
subprocess.run(["python", "-m", "scripts.tok_train"], check=True)
subprocess.run(["python", "-m", "scripts.tok_eval"], check=True)

dataset = Dataset.create(dataset_project=args.project, dataset_name=DATASET_NAME)
dataset.add_files(tokenizer_dir, dataset_path="tokenizer")
dataset.upload()
dataset.finalize()
print(f"Tokenizer dataset ready: {dataset.id}")
