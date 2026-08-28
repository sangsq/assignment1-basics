"""Plot the training and validation loss curves written by cs336_basics.train.

    uv run python scripts/plot_loss.py checkpoints/owt32k --out docs/loss.png

Reads <run>/metrics.jsonl, which holds one JSON object per logged step.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical slots 1 and 2 of a CVD-validated palette (adjacent-pair Delta E 9.1).
TRAIN, VALID = "#2a78d6", "#eb6834"
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"


def load(run: Path) -> tuple[list, list, list, list, list]:
    tr_x, tr_y, va_x, va_y, toks = [], [], [], [], []
    with (run / "metrics.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            if "train_loss" in r:
                tr_x.append(r["step"])
                tr_y.append(r["train_loss"])
                toks.append(r["tokens"])
            elif "valid_loss" in r:
                va_x.append(r["step"])
                va_y.append(r["valid_loss"])
    return tr_x, tr_y, va_x, va_y, toks


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="training output directory")
    p.add_argument("--out", type=Path, default=None, help="defaults to <run>/loss.png")
    p.add_argument("--title", default=None)
    args = p.parse_args()
    out = args.out or args.run / "loss.png"

    tr_x, tr_y, va_x, va_y, toks = load(args.run)
    if not tr_x:
        raise SystemExit(f"no metrics found in {args.run/'metrics.jsonl'}")
    per_step = toks[0] / tr_x[0]
    to_b = lambda s: s * per_step / 1e9  # noqa: E731

    fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot([to_b(s) for s in tr_x], tr_y, color=TRAIN, lw=1, label="train")
    if va_x:
        # Marker so a run with only a couple of eval points still shows something.
        ax.plot([to_b(s) for s in va_x], va_y, color=VALID, lw=2, label="validation",
                marker="o" if len(va_x) < 30 else None, markersize=4)

    ax.set_xlabel("tokens (B)", color=MUTED, fontsize=10)
    ax.set_ylabel("CE loss", color=MUTED, fontsize=10)
    title = args.title or f"OpenWebText — {args.run.name}"
    ax.set_title(title, color=INK, fontsize=13, pad=14, loc="left")

    ax.margins(x=0.04, y=0.08)  # keep end-of-line labels inside the canvas
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)

    # Direct-label the final value of each series; identity is never colour alone.
    if va_x:
        ax.annotate(f"final valid {va_y[-1]:.3f}", (to_b(va_x[-1]), va_y[-1]),
                    textcoords="offset points", xytext=(-6, 10), ha="right",
                    color=INK, fontsize=10, fontweight="bold")
        best = min(va_y)
        # Only worth its own label when the run did not end at its best point.
        if best < va_y[-1] - 5e-4:
            ax.annotate(f"best {best:.3f}", (to_b(va_x[va_y.index(best)]), best),
                        textcoords="offset points", xytext=(-6, -16), ha="right",
                        color=MUTED, fontsize=9)
    leg = ax.legend(frameon=False, loc="upper right", fontsize=10)
    for t in leg.get_texts():
        t.set_color(MUTED)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"{len(tr_x)} train points, {len(va_x)} validation points -> {out}")


if __name__ == "__main__":
    main()
