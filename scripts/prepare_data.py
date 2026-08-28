"""Train a BPE tokenizer on a corpus and encode it to a uint16 .npy token array.

    uv run python scripts/prepare_data.py --name ts \
        --train data/TinyStoriesV2-GPT4-train.txt \
        --valid data/TinyStoriesV2-GPT4-valid.txt

Writes tokenizers/<name>-vocab.json, tokenizers/<name>-merges.txt and
data/<name>_{train,valid}.npy. Encoding streams through the file, so peak
memory stays flat no matter how large the corpus is.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenization import Tokenizer, train_bpe

COPY_STEP = 1 << 26  # tokens per slice when finalising the .npy


def encode_to_npy(tokenizer: Tokenizer, text_path: Path, out_path: Path, chunk_chars: int) -> int:
    """Encode `text_path` into a uint16 .npy of token ids, in bounded memory."""
    tmp = out_path.with_suffix(".tmp")
    total = 0
    t0 = time.perf_counter()
    with open(text_path, encoding="utf-8") as f, open(tmp, "wb") as out:
        carry = ""
        while True:
            chunk = f.read(chunk_chars)
            if not chunk:
                break
            # Read on to the next newline so a pre-token is never split in half.
            tail = f.readline()
            text = carry + chunk + tail
            carry = ""
            ids = np.asarray(tokenizer.encode(text), dtype=np.uint16)
            ids.tofile(out)
            total += ids.size
            print(f"  {out_path.name}: {total/1e6:8.1f}M tokens "
                  f"({total/1e6/(time.perf_counter()-t0):.2f}M tok/s)", end="\r", flush=True)

    # Copy the raw stream into a real .npy now that the length is known.
    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.uint16, shape=(total,))
    src = np.memmap(tmp, dtype=np.uint16, mode="r")
    for i in range(0, total, COPY_STEP):
        arr[i : i + COPY_STEP] = src[i : i + COPY_STEP]
    arr.flush()
    del arr, src
    tmp.unlink()
    print(f"  {out_path.name}: {total/1e6:8.1f}M tokens -> {out_path}          ")
    return total


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="output prefix, e.g. 'ts' or 'owt'")
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--valid", type=Path, required=True)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--tokenizer-dir", type=Path, default=Path("tokenizers"))
    p.add_argument("--processes", type=int, default=None, help="defaults to os.cpu_count()")
    p.add_argument("--chunk-chars", type=int, default=64 << 20, help="text chars per encode call")
    p.add_argument("--reuse-tokenizer", action="store_true", help="load an existing tokenizer instead of training")
    args = p.parse_args()

    args.tokenizer_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.tokenizer_dir / args.name
    vocab_path = prefix.with_name(f"{args.name}-vocab.json")
    merges_path = prefix.with_name(f"{args.name}-merges.txt")

    if args.reuse_tokenizer and vocab_path.exists():
        print(f"loading tokenizer from {vocab_path}")
        tokenizer = Tokenizer.from_files(vocab_path, merges_path, args.special_tokens)
    else:
        print(f"training BPE (vocab={args.vocab_size}) on {args.train}")
        t0 = time.perf_counter()
        vocab, merges = train_bpe(args.train, args.vocab_size, args.special_tokens, args.processes)
        print(f"  {len(vocab)} tokens, {len(merges)} merges in {time.perf_counter()-t0:.1f}s")
        tokenizer = Tokenizer(vocab, merges, args.special_tokens)
        tokenizer.save(prefix)
        print(f"  saved {vocab_path} + {merges_path}")

    for split, path in (("train", args.train), ("valid", args.valid)):
        encode_to_npy(tokenizer, path, args.data_dir / f"{args.name}_{split}.npy", args.chunk_chars)


if __name__ == "__main__":
    main()
