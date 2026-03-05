"""
Training task setup: fetches the tokenizer from the ClearML Dataset created by
clearml_data.py, and wires up shard symlinks from the StorageManager cache.

Called by runs/clearml_run.sh before training starts.
"""
import os

from clearml import Dataset, StorageManager
from nanochat.common import get_base_dir
from nanochat.dataset import BASE_URL, index_to_filename
from scripts.base_eval import EVAL_BUNDLE_URL

DATASET_NAME = "nanochat-tokenizer"
DATASET_PROJECT = os.environ.get("CLEARML_PROJECT", "nanochat")
NUM_SHARDS = int(os.environ.get("NUM_SHARDS", "370"))

base_dir = get_base_dir()
data_dir = os.path.join(base_dir, "base_data")
os.makedirs(data_dir, exist_ok=True)

# -----------------------------------------------------------------------------
# Tokenizer: get from ClearML Dataset, copy into NANOCHAT_BASE_DIR/tokenizer/
dataset = Dataset.get(
    dataset_project=DATASET_PROJECT,
    dataset_name=DATASET_NAME,
    only_completed=True,
)
if dataset is None:
    raise RuntimeError(
        f"Dataset '{DATASET_NAME}' not found in project '{DATASET_PROJECT}'. "
        "Run scripts/clearml_data.py first."
    )
dataset.get_mutable_local_copy(base_dir, overwrite=False)
print(f"Tokenizer ready.")

# -----------------------------------------------------------------------------
# Shards: reuse StorageManager cache populated by the data-prep task.
print(f"Wiring up {NUM_SHARDS} shards...")
for i in range(NUM_SHARDS):
    filename = index_to_filename(i)
    link_path = os.path.join(data_dir, filename)
    if os.path.exists(link_path):
        continue
    cached_path = StorageManager.get_local_copy(f"{BASE_URL}/{filename}")
    os.symlink(cached_path, link_path)
print("Data ready.")

# -----------------------------------------------------------------------------
# Eval bundle: download and extract if not already present
eval_bundle_dir = os.path.join(base_dir, "eval_bundle")
if not os.path.exists(eval_bundle_dir):
    print("Downloading eval bundle...")
    extracted = StorageManager.get_local_copy(EVAL_BUNDLE_URL, extract_archive=True)
    import shutil
    shutil.move(os.path.join(extracted, "eval_bundle"), eval_bundle_dir)
    print(f"Eval bundle ready at {eval_bundle_dir}")
else:
    print("Eval bundle already present.")
