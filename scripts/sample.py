"""Sample text from a trained checkpoint.

    uv run python scripts/sample.py --checkpoint checkpoints/run/best.pt \
        --tokenizer tokenizers/ts --prompt "Once upon a time"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.components import transformerLM
from cs336_basics.decoding import generate
from cs336_basics.tokenization import Tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True, help="prefix of -vocab.json / -merges.txt")
    p.add_argument("--config", type=Path, default=None, help="defaults to <checkpoint dir>/config.json")
    p.add_argument("--prompt", default="Once upon a time")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--num-samples", type=int, default=1)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg_path = args.config or args.checkpoint.parent / "config.json"
    cfg = json.loads(Path(cfg_path).read_text())
    special = ["<|endoftext|>"]

    tokenizer = Tokenizer.from_files(
        args.tokenizer.with_name(f"{args.tokenizer.name}-vocab.json"),
        args.tokenizer.with_name(f"{args.tokenizer.name}-merges.txt"),
        special,
    )

    context_length = int(cfg["context_length"])
    model = transformerLM(
        int(cfg["vocab_size"]), context_length, int(cfg["d_model"]),
        int(cfg["num_layers"]), int(cfg["num_heads"]), int(cfg["d_ff"]), float(cfg["rope_theta"]),
        device=args.device,
    )
    state = torch.load(args.checkpoint, map_location=args.device)
    # Tolerate a checkpoint saved through a torch.compile wrapper.
    weights = {k.removeprefix("_orig_mod."): v for k, v in state["model_state"].items()}
    model.load_state_dict(weights)
    print(f"loaded step {state['iteration']} from {args.checkpoint}\n")

    gen = torch.Generator(device=args.device)
    if args.seed is not None:
        gen.manual_seed(args.seed)

    for i in range(args.num_samples):
        text = generate(
            model, tokenizer, args.prompt,
            max_new_tokens=args.max_new_tokens, context_length=context_length,
            temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
            device=args.device, generator=gen,
        )
        print(f"--- sample {i+1} ---\n{args.prompt}{text}\n")


if __name__ == "__main__":
    main()
