#!/bin/bash
# Single-GPU run for ClearML agent.
# ClearML handles venv/dep installation; this script only does data/training.
set -e

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

python -m nanochat.report reset

# Download data and train tokenizer (skipped if already done on this machine)
python -m nanochat.dataset -n 370
python -m scripts.tok_train
python -m scripts.tok_eval

# Pretraining (single GPU)
python -m scripts.base_train -- ${TRAIN_ARGS:---depth=12 --run=dummy}

python -m nanochat.report generate
