# CS336 Spring 2025 Assignment 1: Basics

A transformer language model implemented from scratch in PyTorch, following the [CS336 Spring 2025 Assignment 1: Basics](https://stanford-cs336.github.io/spring2025/assignments/assignment1_basics.html).

![OpenWebText training curve](docs/owt_loss.png)

92.3M parameters trained on OpenWebText for 4.18B tokens (1.53 epochs) in 9.2
hours on one RTX 4090, reaching a validation loss of **3.433** (perplexity
30.96) at a sustained 126k tokens/s. Train and validation stay within 0.006 of
each other throughout, so the model is still underfitting when the LR schedule
ends.


## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

All 47 tests pass. [./tests/adapters.py](./tests/adapters.py) is the only file
connecting the suite to the implementation in [./cs336_basics/](./cs336_basics/).
`test_train_bpe` checks the learned merges byte-for-byte against a reference
tokenizer, and `test_tokenizer` cross-checks encodings against `tiktoken`.

### Download data
Download a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### Train tokenizer on OWT

Trains a byte-level BPE tokenizer and encodes both splits into `uint16` `.npy`
token arrays:

```sh
uv run python scripts/prepare_data.py --name owt32k \
    --train data/owt_train.txt --valid data/owt_valid.txt \
    --vocab-size 32000
```

Writes `tokenizers/owt32k-{vocab.json,merges.txt}` and
`data/owt32k_{train,valid}.npy`. On 32 cores this takes about 18 minutes for
the 11.9 GB corpus: 50 s to pre-tokenise, 13 min for the 31743 merges, and
~4 min to encode at 16.7M tokens/s. Pre-tokenisation is the part that does not
change when only `--vocab-size` does, so `--pretoken-cache <path>` lets a
retrain skip it.


### Train the model

The run plotted above:

```sh
uv run python -m cs336_basics.train \
    --train data/owt32k_train.npy --valid data/owt32k_valid.npy \
    --vocab-size 32000 --context-length 256 \
    --d-model 704 --num-layers 8 --num-heads 11 --d-ff 1856 \
    --batch-size 96 --max-steps 170000 \
    --lr 3e-4 --min-lr 3e-5 --warmup-steps 2000 \
    --dtype bfloat16 --compile \
    --out checkpoints/owt32k
```

Every flag is listed by `--help`. The run writes its resolved config to
`<out>/config.json`, one JSON record per logged step to `<out>/metrics.jsonl`,
and `best.pt` / `last.pt` checkpoints; `--resume` continues from `last.pt`.
`--compile` is worth about 2.4x on this model.

Then sample from it and plot the curves:

```sh
uv run python scripts/sample.py \
    --checkpoint checkpoints/owt32k/best.pt --tokenizer tokenizers/owt32k \
    --prompt "The future of artificial intelligence"

uv run python scripts/plot_loss.py checkpoints/owt32k --out docs/owt_loss.png
```