#!/bin/bash
# Training task for ClearML agent (single GPU).
# Requires scripts/clearml_data.py to have been run at least once.
set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

python -m nanochat.report reset

# Fetch tokenizer from ClearML Dataset and wire up shard symlinks
python -m scripts.clearml_setup

# Pretraining (single GPU)
python -m scripts.base_train ${TRAIN_ARGS:---depth=12}

python -c "
from clearml import Task
from nanochat.report import get_report
report_file = get_report().generate()
task = Task.current_task()
if task:
    task.upload_artifact('report', report_file)
"
