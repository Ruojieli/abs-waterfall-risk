"""Figures and Markdown tables for the reports folder."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
DIVERGING = LinearSegmentedColormap.from_list(
    "red_gray_blue", ["#9e2a2b", "#e34948", "#f0efec", "#6da7ec", "#1c5cab"])

plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False})


def md_table(df: pd.DataFrame, floatfmt: str = "{:,.2f}") -> str:
    """GitHub Markdown table (no tabulate dependency)."""
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    def fmt(v):
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] + ["---:"] * len(df.columns)) + "|"]
    for idx, *vals in df.itertuples():
        lines.append("| " + " | ".join([str(idx)] + [fmt(v) for v in vals]) + " |")
    return "\n".join(lines)


def day_heatmap(grid: pd.DataFrame, path: Path) -> None:
    """Duration-adjusted yield by prepay x loss stress, one panel per class."""
    classes = list(dict.fromkeys(grid["class"]))
    vals = 100 * grid["dur_adj_yield"]
    norm = TwoSlopeNorm(vcenter=0.0, vmin=min(vals.min(), -1.0), vmax=vals.max())
    fig, axes = plt.subplots(1, len(classes), figsize=(3.3 * len(classes), 3.4), sharey=True)
    for ax, c in zip(np.atleast_1d(axes), classes):
        g = grid[grid["class"] == c].pivot(index="prepay_mult", columns="loss_mult",
                                           values="dur_adj_yield") * 100
        ax.imshow(g.values, cmap=DIVERGING, norm=norm, aspect="auto", origin="lower")
        for (i, j), v in np.ndenumerate(g.values):
            dark = abs(norm(v) - 0.5) > 0.32
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if dark else INK)
        ax.set_xticks(range(g.shape[1]), [f"{x:g}x" for x in g.columns])
        ax.set_yticks(range(g.shape[0]), [f"{y:g}x" for y in g.index])
        ax.set_title(f"Class {c}", fontsize=10, color=INK)
        ax.set_xlabel("Loss stress (x base CDR)")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    np.atleast_1d(axes)[0].set_ylabel("Prepay stress (x base ABS speed)")
    fig.suptitle("Duration-adjusted yield (yield / modified duration, %) at the base-case price",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def paydown_chart(res, deal, path: Path) -> None:
    """Stacked class balances (plus the OC cushion) over the life of the deal."""
    months = np.r_[0, res.collateral.index.to_numpy()]
    stacks = [np.r_[t.balance, res.classes[t.name]["end_balance"]] / 1e6 for t in deal.tranches]
    stacks.append(np.r_[deal.pool.total - sum(t.balance for t in deal.tranches), res.deal["oc"]] / 1e6)
    labels = [f"Class {n}" for n in deal.names] + ["OC (equity)"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.stackplot(months, stacks, labels=labels, colors=SERIES[:len(stacks)],
                 edgecolor=SURFACE, linewidth=1.0)
    ax.set_xlabel("Payment month")
    ax.set_ylabel("Balance ($mm)")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.set_title(f"{deal.name}: base-case paydown (clean-up call at month {months[-1]})",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def speeds_chart(stats: pd.DataFrame, path: Path) -> None:
    """Realized monthly prepayment and default speeds from the loan tapes."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
    x = pd.to_datetime(stats["period"]).dt.strftime("%b")
    for ax, col, title, color in [(axes[0], "cpr", "Prepayment, CPR (%)", SERIES[0]),
                                  (axes[1], "cdr", "Gross default, CDR (%)", SERIES[1])]:
        ax.plot(x, 100 * stats[col], color=color, linewidth=2, marker="o", markersize=5)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_ylim(bottom=0)
    fig.suptitle("FORDO 2026-A realized speeds, collection periods Mar-Sep 2026 (ABS-EE)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
