"""Byte-level BPE: training, encoding, and GPT-2-style (de)serialisation."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from heapq import heapify, heappop, heappush
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries

# GPT-2's pre-tokenisation pattern: splits on word/number/punctuation/whitespace runs
# so that merges can never span a word boundary.
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def gpt2_bytes_to_unicode() -> dict[int, str]:
    """Map each of the 256 bytes to a printable unicode char, as GPT-2 does.

    Used only for serialisation, so that `vocab.json` / `merges.txt` stay
    human-readable and interchangeable with the reference GPT-2 files.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs), strict=True))


# --------------------------------------------------------------------------- #
# Pre-tokenisation
# --------------------------------------------------------------------------- #


def pre_tokenization(text: str, PAT: str, special_tokens: list[str], pre_tokens_str=None) -> dict[str, int]:
    """Count pre-tokens in `text`, never merging across a special token."""
    i = 0
    if pre_tokens_str is None:
        pre_tokens_str = defaultdict(int)
    if special_tokens:
        st_PAT = "|".join(re.escape(x) for x in special_tokens)
        for st_match in re.finditer(st_PAT, text):
            j = st_match.start()
            for match in re.finditer(PAT, text[i:j]):
                pre_tokens_str[match.group()] += 1
            i = st_match.end()
    for match in re.finditer(PAT, text[i:]):
        pre_tokens_str[match.group()] += 1
    return pre_tokens_str


def _pretokenize_chunk(args: tuple[str, int, int, list[str]]) -> dict[str, int]:
    """Worker: read one byte range of a file and count its pre-tokens."""
    path, start, end, special_tokens = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return dict(pre_tokenization(text, PAT, special_tokens))


def pretokenize_file(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> dict[str, int]:
    """Pre-tokenise a corpus in parallel without ever holding it all in memory.

    The file is split on `special_tokens[0]` boundaries so no pre-token — and
    therefore no merge — can straddle two chunks.
    """
    num_processes = num_processes or (os.cpu_count() or 1)
    split_token = (special_tokens[0] if special_tokens else "\n").encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes * 8, split_token)

    jobs = [(str(input_path), s, e, special_tokens) for s, e in zip(boundaries[:-1], boundaries[1:], strict=True)]
    counts: dict[str, int] = defaultdict(int)
    with Pool(num_processes) as pool:
        for partial in pool.imap_unordered(_pretokenize_chunk, jobs):
            for word, c in partial.items():
                counts[word] += c
    return counts


_ENCODER: Tokenizer | None = None


def _encode_init(vocab, merges, special_tokens) -> None:
    global _ENCODER
    _ENCODER = Tokenizer(vocab, merges, special_tokens)


def _encode_chunk(args: tuple[str, int, int]) -> np.ndarray:
    """Worker: encode one byte range of a file to a uint16 array of token ids."""
    path, start, end = args
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return np.asarray(_ENCODER.encode(text), dtype=np.uint16)


def encode_file_chunks(
    tokenizer: Tokenizer,
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
    chunks_per_process: int = 8,
):
    """Yield uint16 arrays of token ids, in file order, encoded in parallel.

    Chunks are cut on special-token boundaries, so concatenating the results
    reproduces exactly what encoding the whole file in one call would give.
    """
    num_processes = num_processes or (os.cpu_count() or 1)
    split_token = (special_tokens[0] if special_tokens else "\n").encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes * chunks_per_process, split_token)
    jobs = [(str(input_path), s, e) for s, e in zip(boundaries[:-1], boundaries[1:], strict=True)]

    merges = [m for m, _ in sorted(tokenizer.merges.items(), key=lambda kv: kv[1])]
    with Pool(
        num_processes,
        initializer=_encode_init,
        initargs=(tokenizer.id2token, merges, special_tokens),
    ) as pool:
        yield from pool.imap(_encode_chunk, jobs)  # imap preserves order


# --------------------------------------------------------------------------- #
# BPE training
# --------------------------------------------------------------------------- #


class _MaxPair:
    """Heap key that orders byte-pairs in *descending* lexicographic order.

    heapq is a min-heap, but ties on frequency must be broken by taking the
    lexicographically greatest pair, so the comparison is inverted here.
    """

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: _MaxPair) -> bool:
        return self.pair > other.pair


def construct_bpe(
    pre_tokens_str: dict[str, int],
    vocab_size: int,
    special_tokens: list[str],
    progress: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Learn BPE merges from pre-token counts.

    Each merge only touches the words that actually contain the merged pair
    (tracked in an inverted index) and the next pair is pulled from a lazily
    updated max-heap, so a merge costs O(affected words) rather than a full
    scan of the corpus.
    """
    words: list[list[bytes]] = []
    freqs: list[int] = []
    for word, count in pre_tokens_str.items():
        words.append([bytes([b]) for b in word.encode("utf-8")])
        freqs.append(count)

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    token_id = 256
    for token in special_tokens or []:
        vocab[token_id] = token.encode("utf-8")
        token_id += 1

    # pair -> total frequency, and pair -> indices of words containing it
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for wi, word in enumerate(words):
        for pair in zip(word, word[1:], strict=False):
            pair_counts[pair] += freqs[wi]
            pair_words[pair].add(wi)

    heap = [(-c, _MaxPair(p), p) for p, c in pair_counts.items()]
    heapify(heap)

    merges: list[tuple[bytes, bytes]] = []
    bar = None
    if progress:
        from tqdm.auto import tqdm

        bar = tqdm(total=vocab_size - len(vocab), desc="merges", unit="merge")
    while len(vocab) < vocab_size:
        # Pop until we find an entry whose count is still current (lazy deletion).
        best = None
        while heap:
            neg_count, _, pair = heappop(heap)
            if -neg_count > 0 and pair_counts.get(pair, 0) == -neg_count:
                best = pair
                break
        if best is None:
            break  # corpus fully merged

        left, right = best
        merged = left + right
        merges.append(best)
        vocab[token_id] = merged
        token_id += 1

        for wi in pair_words[best]:
            word = words[wi]
            freq = freqs[wi]
            i = 0
            while i < len(word) - 1:
                if word[i] != left or word[i + 1] != right:
                    i += 1
                    continue
                # Only the two pairs straddling the merge site change. Splice in
                # place so the left neighbour already reflects earlier merges in
                # this same word.
                if i > 0:
                    prev = word[i - 1]
                    pair_counts[prev, left] -= freq
                    heappush(heap, (-pair_counts[prev, left], _MaxPair((prev, left)), (prev, left)))
                    pair_counts[prev, merged] += freq
                    heappush(heap, (-pair_counts[prev, merged], _MaxPair((prev, merged)), (prev, merged)))
                    pair_words[prev, merged].add(wi)
                if i + 2 < len(word):
                    nxt = word[i + 2]
                    pair_counts[right, nxt] -= freq
                    heappush(heap, (-pair_counts[right, nxt], _MaxPair((right, nxt)), (right, nxt)))
                    pair_counts[merged, nxt] += freq
                    heappush(heap, (-pair_counts[merged, nxt], _MaxPair((merged, nxt)), (merged, nxt)))
                    pair_words[merged, nxt].add(wi)
                pair_counts[best] -= freq
                word[i : i + 2] = (merged,)
                i += 1

        pair_counts.pop(best, None)
        pair_words.pop(best, None)
        if bar is not None:
            bar.update(1)

    if bar is not None:
        bar.close()
    return vocab, merges


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
    progress: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer straight from a corpus on disk."""
    pre_tokens = pretokenize_file(input_path, special_tokens, num_processes)
    return construct_bpe(pre_tokens, vocab_size, special_tokens, progress=progress)


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens, PAT=None):
        # bidirectional mapping between ids and tokens
        self.id2token = vocab
        self.token2id = {token: i for i, token in vocab.items()}

        # merge -> rank, so the lowest-ranked applicable merge is a dict lookup
        self.merges = {merge: rank for rank, merge in enumerate(merges)}

        if special_tokens:
            # Copy before sorting: an in-place .sort() would reorder the caller's list.
            # Longest-first so that overlapping specials match greedily.
            special_tokens = sorted(special_tokens, key=len, reverse=True)
            self.st_PAT = "|".join(re.escape(x) for x in special_tokens)
        else:
            self.st_PAT = None
        self.special_tokens = special_tokens
        self.PAT = PAT or globals()["PAT"]

    # -- serialisation ------------------------------------------------------ #

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None, PAT=None) -> Tokenizer:
        """Load a tokenizer from GPT-2-style `vocab.json` + `merges.txt`."""
        decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        def to_bytes(s: str) -> bytes:
            return bytes(decoder[ch] for ch in s)

        with open(vocab_path, encoding="utf-8") as f:
            vocab = {idx: to_bytes(tok) for tok, idx in json.load(f).items()}
        merges = []
        with open(merges_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) == 2:
                    merges.append((to_bytes(parts[0]), to_bytes(parts[1])))

        for token in special_tokens or []:
            if token.encode("utf-8") not in set(vocab.values()):
                vocab[len(vocab)] = token.encode("utf-8")
        return cls(vocab, merges, special_tokens, PAT)

    def save(self, prefix: str | os.PathLike) -> tuple[Path, Path]:
        """Write `<prefix>-vocab.json` and `<prefix>-merges.txt`.

        Plain text rather than a pickled object: reloading a tokenizer must not
        require executing whatever was serialised alongside it.
        """
        encoder = gpt2_bytes_to_unicode()

        def to_str(b: bytes) -> str:
            return "".join(encoder[x] for x in b)

        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        vocab_path = prefix.with_name(prefix.name + "-vocab.json")
        merges_path = prefix.with_name(prefix.name + "-merges.txt")

        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump({to_str(tok): idx for idx, tok in self.id2token.items()}, f, ensure_ascii=False)
        inverse = sorted(self.merges.items(), key=lambda kv: kv[1])
        with open(merges_path, "w", encoding="utf-8") as f:
            f.writelines(f"{to_str(a)} {to_str(b)}\n" for (a, b), _ in inverse)
        return vocab_path, merges_path

    # -- encoding ----------------------------------------------------------- #

    def _encode_pretoken(self, bs: bytes) -> list[int]:
        if bs in self.token2id:
            return [self.token2id[bs]]

        word = [bytes([b]) for b in bs]
        while len(word) > 1:
            # Single pass for the lowest-ranked merge; no per-iteration allocation,
            # which matters because encode_iterable runs under a tight memory cap.
            best_rank, best_i = None, -1
            for i in range(len(word) - 1):
                rank = self.merges.get((word[i], word[i + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_i = rank, i
            if best_i < 0:
                break
            word[best_i : best_i + 2] = [word[best_i] + word[best_i + 1]]

        return [self.token2id[t] for t in word]

    def encode(self, s: str) -> list[int]:
        result: list[int] = []
        i = 0
        if self.special_tokens:
            for st_match in re.finditer(self.st_PAT, s):
                for match in re.finditer(self.PAT, s[i : st_match.start()]):
                    result.extend(self._encode_pretoken(match.group().encode("utf-8")))
                result.append(self.token2id[st_match.group().encode("utf-8")])
                i = st_match.end()
        for match in re.finditer(self.PAT, s[i:]):
            result.extend(self._encode_pretoken(match.group().encode("utf-8")))
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Stream token ids from an iterable of strings (e.g. an open file)."""
        for s in iterable:
            yield from self.encode(s)

    def decode(self, token_ids: Iterable[int]) -> str:
        return b"".join(self.id2token[i] for i in token_ids).decode("utf-8", errors="replace")
