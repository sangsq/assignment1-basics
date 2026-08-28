"""Train the Transformer LM from the command line.

    uv run python -m cs336_basics.train --train data/ts_train.npy --valid data/ts_valid.npy

Every knob is a flag, the resolved config is written next to the checkpoints,
and `--resume` picks a run back up from its last checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.components import (
    AdamW,
    cross_entropy_loss,
    data_loader,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
    transformerLM,
)

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    d = p.add_argument_group("data")
    d.add_argument("--train", type=Path, required=True, help="uint16 .npy of training token ids")
    d.add_argument("--valid", type=Path, required=True, help="uint16 .npy of validation token ids")

    m = p.add_argument_group("model")
    m.add_argument("--vocab-size", type=int, default=10000)
    m.add_argument("--context-length", type=int, default=256)
    m.add_argument("--d-model", type=int, default=512)
    m.add_argument("--num-layers", type=int, default=4)
    m.add_argument("--num-heads", type=int, default=16)
    m.add_argument("--d-ff", type=int, default=1344, help="defaults to ~8/3 * d_model rounded to 64")
    m.add_argument("--rope-theta", type=float, default=10000.0)

    o = p.add_argument_group("optimisation")
    o.add_argument("--batch-size", type=int, default=128)
    o.add_argument("--max-steps", type=int, default=20000)
    o.add_argument("--lr", type=float, default=3e-4, help="peak learning rate")
    o.add_argument("--min-lr", type=float, default=3e-5)
    o.add_argument("--warmup-steps", type=int, default=500)
    o.add_argument("--cosine-steps", type=int, default=None, help="defaults to --max-steps")
    o.add_argument("--weight-decay", type=float, default=0.1)
    o.add_argument("--beta1", type=float, default=0.9)
    o.add_argument("--beta2", type=float, default=0.95)
    o.add_argument("--eps", type=float, default=1e-8)
    o.add_argument("--grad-clip", type=float, default=1.0, help="0 disables clipping")

    r = p.add_argument_group("runtime")
    r.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    r.add_argument("--dtype", choices=list(DTYPES), default="bfloat16", help="autocast dtype")
    r.add_argument("--compile", action="store_true", help="wrap the model in torch.compile")
    r.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("logging & checkpoints")
    g.add_argument("--out", type=Path, default=Path("checkpoints/run"))
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--eval-every", type=int, default=500)
    g.add_argument("--eval-batches", type=int, default=20)
    g.add_argument("--save-every", type=int, default=2000)
    g.add_argument("--resume", action="store_true", help="continue from <out>/last.pt if present")

    return p.parse_args(argv)


@torch.no_grad()
def evaluate(net, data, args, autocast) -> float:
    """Mean validation loss over a fixed set of windows (same seed every call)."""
    net.eval()
    rng = np.random.default_rng(args.seed)  # fixed -> eval curves are comparable across steps
    total = 0.0
    for _ in range(args.eval_batches):
        x, y = data_loader(data, args.context_length, args.batch_size, args.device, rng=rng)
        with autocast:
            total += cross_entropy_loss(net(x), y).item()
    net.train()
    return total / args.eval_batches


def main(argv=None) -> None:
    args = parse_args(argv)
    args.cosine_steps = args.cosine_steps or args.max_steps
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # TF32 for the fp32 matmuls that autocast leaves alone.
    torch.set_float32_matmul_precision("high")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    train_data = np.load(args.train, mmap_mode="r")  # memmap: the corpus never enters RAM
    valid_data = np.load(args.valid, mmap_mode="r")
    print(f"train {len(train_data)/1e6:.1f}M tokens | valid {len(valid_data)/1e6:.1f}M tokens")

    model = transformerLM(
        args.vocab_size, args.context_length, args.d_model,
        args.num_layers, args.num_heads, args.d_ff, args.rope_theta,
        device=args.device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params/1e6:.1f}M parameters")
    # `net` is what runs; `model` stays the bare module. torch.compile's wrapper
    # prefixes every state_dict key with "_orig_mod.", so checkpointing through it
    # would produce files nothing else can load.
    net = torch.compile(model) if args.compile else model

    optimizer = AdamW(
        model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay, eps=args.eps,
    )

    start_step = 0
    if args.resume and (args.out / "last.pt").exists():
        start_step = load_checkpoint(args.out / "last.pt", model, optimizer)
        print(f"resumed from step {start_step}")

    use_autocast = args.device.startswith("cuda") and args.dtype != "float32"
    autocast = (
        torch.autocast(device_type="cuda", dtype=DTYPES[args.dtype]) if use_autocast
        else torch.autocast(device_type="cpu", enabled=False)
    )

    log_path = args.out / "metrics.jsonl"
    best_val = math.inf
    net.train()
    t0 = time.perf_counter()
    tokens_seen = start_step * args.batch_size * args.context_length

    for step in range(start_step, args.max_steps):
        lr = get_lr_cosine_schedule(step, args.lr, args.min_lr, args.warmup_steps, args.cosine_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = data_loader(train_data, args.context_length, args.batch_size, args.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast:
            loss = cross_entropy_loss(net(x), y)
        loss.backward()
        # Clip *after* backward: the original notebook clipped before, so the
        # gradients were still None and nothing was ever scaled.
        if args.grad_clip:
            gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        tokens_seen += args.batch_size * args.context_length
        step += 1

        if step % args.log_every == 0:
            elapsed = time.perf_counter() - t0
            rec = {
                "step": step, "train_loss": loss.item(), "lr": lr,
                "tokens": tokens_seen, "tokens_per_sec": tokens_seen / elapsed,
            }
            print(f"step {step:6d} | loss {rec['train_loss']:.4f} | lr {lr:.2e} | "
                  f"{rec['tokens_per_sec']/1e3:.0f}k tok/s")
            with log_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")

        if step % args.eval_every == 0 or step == args.max_steps:
            val = evaluate(net, valid_data, args, autocast)
            print(f"step {step:6d} | valid loss {val:.4f}")
            with log_path.open("a") as f:
                f.write(json.dumps({"step": step, "valid_loss": val}) + "\n")
            if val < best_val:
                best_val = val
                save_checkpoint(model, optimizer, step, args.out / "best.pt")

        if step % args.save_every == 0 or step == args.max_steps:
            save_checkpoint(model, optimizer, step, args.out / "last.pt")

    save_checkpoint(model, optimizer, args.max_steps, args.out / "last.pt")
    print(f"done. best valid loss {best_val:.4f} -> {args.out/'best.pt'}")


if __name__ == "__main__":
    main()
