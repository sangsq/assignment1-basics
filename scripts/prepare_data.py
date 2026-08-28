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
import pickle
import time
from pathlib import Path

import numpy as np

from cs336_basics.tokenization import (
    Tokenizer,
    construct_bpe,
    encode_file_chunks,
    pretokenize_file,
)

COPY_STEP = 1 << 26  # tokens per slice when finalising the .npy


def encode_to_npy(tokenizer: Tokenizer, text_path: Path, out_path: Path,
                  special_tokens: list[str], processes: int | None) -> int:
    """Encode `text_path` into a uint16 .npy of token ids, in bounded memory.

    Chunks are encoded in parallel and written in file order, so the result is
    identical to encoding the whole file in one call.
    """
    tmp = out_path.with_suffix(".tmp")
    total = 0
    t0 = time.perf_counter()
    with open(tmp, "wb") as out:
        for ids in encode_file_chunks(tokenizer, text_path, special_tokens, processes):
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
    p.add_argument("--reuse-tokenizer", action="store_true", help="load an existing tokenizer instead of training")
    p.add_argument("--pretoken-cache", type=Path, default=None,
                   help="pickle of pre-token counts; reused if present, written if not. "
                        "Pre-tokenising a large corpus is the slow part of a retrain.")
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
        print(f"training BPE (vocab={args.vocab_size}) on {args.train}", flush=True)
        t0 = time.perf_counter()
        if args.pretoken_cache and args.pretoken_cache.exists():
            print(f"  reusing pre-token counts from {args.pretoken_cache}", flush=True)
            with open(args.pretoken_cache, "rb") as f:
                pre_tokens = pickle.load(f)
        else:
            pre_tokens = pretokenize_file(args.train, args.special_tokens, args.processes)
            if args.pretoken_cache:
                with open(args.pretoken_cache, "wb") as f:
                    pickle.dump(pre_tokens, f, protocol=5)
        print(f"  {len(pre_tokens):,} distinct pre-tokens ({time.perf_counter()-t0:.0f}s)", flush=True)
        vocab, merges = construct_bpe(pre_tokens, args.vocab_size, args.special_tokens, progress=True)
        print(f"  {len(vocab)} tokens, {len(merges)} merges in {time.perf_counter()-t0:.1f}s")
        tokenizer = Tokenizer(vocab, merges, args.special_tokens)
        tokenizer.save(prefix)
        print(f"  saved {vocab_path} + {merges_path}")

    for split, path in (("train", args.train), ("valid", args.valid)):
        encode_to_npy(tokenizer, path, args.data_dir / f"{args.name}_{split}.npy",
                      args.special_tokens, args.processes)


if __name__ == "__main__":
    main()
