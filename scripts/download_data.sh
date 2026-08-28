#!/usr/bin/env bash
# Fetch the TinyStories and OpenWebText-sample corpora into ./data.
# The files are large (TinyStories ~2.2 GB, OWT ~12 GB) and are gitignored;
# set DATA_DIR to keep them on another volume and symlink ./data at it.
set -euo pipefail

DATA_DIR="${DATA_DIR:-$(dirname "$0")/../data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "downloading into $(pwd)"

TS=https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main
wget -nc "$TS/TinyStoriesV2-GPT4-train.txt"
wget -nc "$TS/TinyStoriesV2-GPT4-valid.txt"

OWT=https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main
wget -nc "$OWT/owt_train.txt.gz" && gunzip -kf owt_train.txt.gz
wget -nc "$OWT/owt_valid.txt.gz" && gunzip -kf owt_valid.txt.gz

echo "done:"
ls -lh
