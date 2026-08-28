"""Autoregressive sampling: temperature, top-k, and nucleus (top-p) truncation."""

from __future__ import annotations

import torch

from cs336_basics.components import softmax


def _truncate(probs: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    """Zero out the tail of the distribution, then renormalise."""
    if top_k:
        kth = probs.topk(min(top_k, probs.size(-1)), dim=-1).values[..., -1:]
        probs = probs.masked_fill(probs < kth, 0.0)
    if top_p is not None and top_p < 1.0:
        ordered, idx = probs.sort(dim=-1, descending=True)
        cumulative = ordered.cumsum(dim=-1)
        # Keep everything up to and including the token that crosses `top_p`,
        # so the smallest nucleus is always at least one token.
        drop = cumulative - ordered > top_p
        ordered = ordered.masked_fill(drop, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, idx, ordered)
    return probs / probs.sum(dim=-1, keepdim=True)


@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    context_length: int | None = None,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = 0.95,
    eos_token: str | None = "<|endoftext|>",
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> str:
    """Sample a continuation of `prompt` and return it as decoded text.

    Stops early on `eos_token`. `context_length` caps how much history is fed
    back in, which must not exceed what the model's RoPE tables were built for.
    """
    model.eval()
    ids = tokenizer.encode(prompt)
    eos_id = tokenizer.token2id.get(eos_token.encode("utf-8")) if eos_token else None

    generated: list[int] = []
    for _ in range(max_new_tokens):
        window = (ids + generated)[-context_length:] if context_length else ids + generated
        x = torch.tensor(window, dtype=torch.long, device=device)
        logits = model(x)[-1, :].float()

        if temperature <= 0:  # greedy
            next_id = int(logits.argmax())
        else:
            probs = _truncate(softmax(logits / temperature, dim=-1), top_k, top_p)
            # torch.multinomial requires the generator to live on the same device
            # as the probabilities; honour a caller's CPU generator either way.
            if generator is not None and generator.device != probs.device:
                probs = probs.to(generator.device)
            next_id = int(torch.multinomial(probs, num_samples=1, generator=generator))

        if eos_id is not None and next_id == eos_id:
            break
        generated.append(next_id)

    return tokenizer.decode(generated)
