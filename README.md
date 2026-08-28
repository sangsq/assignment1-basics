# A Transformer LM from scratch

[![CI](https://github.com/sangsq/assignment1-basics/actions/workflows/ci.yml/badge.svg)](https://github.com/sangsq/assignment1-basics/actions/workflows/ci.yml)

A byte-level BPE tokenizer and a decoder-only Transformer language model, both written
from primitives — no `nn.Linear`, no `nn.MultiheadAttention`, no `F.scaled_dot_product_attention`,
no `torch.optim.AdamW`. Built against the
[Stanford CS336](https://stanford-cs336.github.io/spring2025/) Spring 2025 Assignment 1
test suite ([handout](https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_spring2025_assignment1_basics.pdf)).

```sh
uv run pytest        # 47 passed, 1 xpassed
```

## What's implemented

| Component | Where | Notes |
|---|---|---|
| Byte-level BPE training | [tokenization.py](cs336_basics/tokenization.py) | inverted index + lazy-deletion max-heap; parallel chunked pre-tokenization |
| BPE encode / decode | [tokenization.py](cs336_basics/tokenization.py) | streaming `encode_iterable`, GPT-2-format `vocab.json` / `merges.txt` |
| Linear, Embedding, RMSNorm, SwiGLU | [components.py](cs336_basics/components.py) | truncated-normal init, fp32 norm accumulation |
| RoPE | [components.py](cs336_basics/components.py) | precomputed cos/sin tables, fully vectorized |
| Causal multi-head attention | [components.py](cs336_basics/components.py) | batched across heads, hand-written softmax |
| Transformer blocks + LM head | [components.py](cs336_basics/components.py) | pre-norm residual stream |
| Cross-entropy, AdamW, cosine LR, grad clipping | [components.py](cs336_basics/components.py) | numerically-stable logsumexp |
| Sampling | [decoding.py](cs336_basics/decoding.py) | temperature, top-k, nucleus (top-p), EOS stop |
| Training loop | [train.py](cs336_basics/train.py) | memmapped data, bf16 autocast, `torch.compile`, checkpoint/resume, W&B |
| PyTorch reference build | [torch_ver.py](cs336_basics/torch_ver.py) | same architecture on stock modules, kept as a correctness reference |

## Quickstart

```sh
# 1. environment (https://docs.astral.sh/uv/)
uv sync

# 2. corpora -> ./data  (TinyStories ~2.2 GB; set DATA_DIR to put them elsewhere)
./scripts/download_data.sh

# 3. train a BPE tokenizer and encode the corpus to uint16 .npy
uv run python scripts/prepare_data.py --name ts \
    --train data/TinyStoriesV2-GPT4-train.txt \
    --valid data/TinyStoriesV2-GPT4-valid.txt \
    --vocab-size 10000

# 4. train
uv run python -m cs336_basics.train \
    --train data/ts_train.npy --valid data/ts_valid.npy \
    --vocab-size 10000 --context-length 256 \
    --d-model 512 --num-layers 4 --num-heads 16 --d-ff 1344 \
    --batch-size 128 --max-steps 20000 --lr 3e-4 --compile \
    --out checkpoints/ts

# 5. sample
uv run python scripts/sample.py \
    --checkpoint checkpoints/ts/best.pt --tokenizer tokenizers/ts \
    --prompt "Once upon a time"

# 6. plot the loss curves
uv run python scripts/plot_loss.py checkpoints/ts
```

`--resume` continues from `<out>/last.pt`, `--compile` is worth about 2.4x, and every
flag is listed by `--help`. Each run writes its resolved config to
`<out>/config.json` and one JSON record per logged step to `<out>/metrics.jsonl`,
next to the `best.pt` / `last.pt` checkpoints.

OpenWebText is the same pipeline with `--vocab-size 32000` (what the assignment
asks for on that corpus) and a larger model.

## Implementation notes

Two rewrites carried most of the performance work, both verified numerically
identical to the straightforward versions they replaced:

- **Attention and RoPE are batched, not looped.** The first version ran a Python
  loop over attention heads, and a second loop over RoPE's `d_k/2` channel pairs
  applying an explicit 2×2 rotation to each. Both now collapse into batched tensor
  ops — heads folded into a leading axis, RoPE reduced to two elementwise cos/sin
  multiplies against precomputed tables. Roughly **4.5×** faster per fwd+bwd step.
- **BPE training uses an inverted index.** Each merge used to rescan the whole
  corpus and pick the next pair with a full `max()` over the frequency table. It now
  keeps a byte-pair → containing-words index and a lazily-updated max-heap, so a
  merge costs time proportional to the words it actually touches. About **25×**
  faster at vocab 8000, and unlike the old version the cost barely grows with
  vocabulary size.

Pre-tokenization runs in parallel across processes over special-token-aligned file
chunks, and encoding streams through the corpus, so neither step needs the corpus
in memory.

## Results

_Pending — a full TinyStories training run has not been done yet._

## Layout

```
cs336_basics/
  tokenization.py   BPE training, encoding, serialisation
  components.py     model + optimiser primitives
  decoding.py       sampling
  train.py          training CLI
  torch_ver.py      same architecture on stock PyTorch modules, as a reference
scripts/
  download_data.sh  fetch TinyStories / OpenWebText
  prepare_data.py   train tokenizer, encode corpus to .npy
  sample.py         generate from a checkpoint
  plot_loss.py      loss curves from a run's metrics.jsonl
notebooks/          exploratory analysis (OpenWebText pre-token statistics)
tests/              CS336 test suite; adapters.py wires it to this implementation
```

Corpora, encoded token arrays, trained tokenizers and checkpoints are all gitignored —
they are reproduced from `scripts/`. Tokenizers serialise to plain
`vocab.json` + `merges.txt` rather than pickles, so loading one never means executing
whatever was stored alongside it.

## Tests

```sh
uv run pytest                   # full suite
uvx ruff check cs336_basics scripts tests
```

`tests/` is the unmodified CS336 suite; [tests/adapters.py](tests/adapters.py) is the
only file that connects it to this implementation. `test_train_bpe` checks the learned
merges byte-for-byte against a reference tokenizer, and `test_tokenizer` cross-checks
encodings against `tiktoken`.

## Attribution

Starter code, tests, and fixtures come from
[stanford-cs336/assignment1-basics](https://github.com/stanford-cs336/assignment1-basics)
(MIT, see [LICENSE](LICENSE)). Everything under `cs336_basics/` apart from
`pretokenization_example.py`, plus everything under `scripts/`, is my own work.
