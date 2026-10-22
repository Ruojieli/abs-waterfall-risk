"""Tranche analytics from projected monthly cash flows.

Yields are bond-equivalent (semiannual compounding) on monthly cash flows: a cash
flow in month m is discounted by (1 + y/2)^(-m/6). Prices are per 100 of current
face, with settlement on a payment date (no accrued interest).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from .collateral import Assumptions
from .curve import Curve
from .waterfall import Deal, WaterfallResult, run_waterfall


def _t(n: int) -> np.ndarray:
    return np.arange(1, n + 1) / 12


def price_from_yield(cf: np.ndarray, y: float, face: float) -> float:
    return float(100 * cf @ (1 + y / 2) ** (-2 * _t(len(cf))) / face)


def yield_from_price(cf: np.ndarray, price: float, face: float) -> float:
    """Cash-flow yield, floored at -100% for classes that lose most of their value."""
    f = lambda y: price_from_yield(cf, y, face) - price
    if f(-1.0) <= 0:
        return -1.0
    return float(brentq(f, -1.0, 10.0, xtol=1e-12))


def wal(principal: np.ndarray) -> float:
    return float(_t(len(principal)) @ principal / principal.sum())


def window(principal: np.ndarray) -> tuple[int, int]:
    """First and last month with a principal payment."""
    paid = np.flatnonzero(principal > 1e-6 * principal.sum())
    return int(paid[0]) + 1, int(paid[-1]) + 1


def macaulay(cf: np.ndarray, y: float) -> float:
    t = _t(len(cf))
    pv = cf * (1 + y / 2) ** (-2 * t)
    return float(t @ pv / pv.sum())


def modified_duration(cf: np.ndarray, y: float) -> float:
    return macaulay(cf, y) / (1 + y / 2)


def convexity(cf: np.ndarray, y: float) -> float:
    t = _t(len(cf))
    pv = cf * (1 + y / 2) ** (-2 * t)
    return float((t * (t + 0.5)) @ pv / pv.sum() / (1 + y / 2) ** 2)


def price_on_curve(cf: np.ndarray, face: float, curve: Curve, z: float) -> float:
    """Price discounting each cash flow on the zero curve plus a constant Z-spread."""
    return float(100 * cf @ curve.discount(_t(len(cf)), z) / face)


def z_spread(cf: np.ndarray, price: float, face: float, curve: Curve) -> float:
    return float(brentq(lambda z: price_on_curve(cf, face, curve, z) - price, -0.5, 5.0,
                        xtol=1e-12))


def metrics(cf: np.ndarray, principal: np.ndarray, price: float, face: float,
            curve: Curve | None = None) -> dict:
    y = yield_from_price(cf, price, face)
    d = modified_duration(cf, y)
    out = {"price": price, "yield": y, "wal": wal(principal), "window": window(principal),
           "mac_dur": macaulay(cf, y), "mod_dur": d, "convexity": convexity(cf, y),
           "dur_adj_yield": y / d}
    if curve is not None:
        out["i_spread"] = y - float(curve.par_yield(out["wal"]))
    return out


def price_at_spread(res: WaterfallResult, name: str, face: float, spread: float,
                    curve: Curve) -> float:
    """Price at a spread over the interpolated Treasury at the tranche's WAL."""
    y = float(curve.par_yield(wal(res.principal(name)))) + spread
    return price_from_yield(res.cash_flow(name), y, face)


def _fmt_window(w: tuple[int, int], first_pay: pd.Timestamp | None) -> str:
    if first_pay is None:
        return f"{w[0]}-{w[1]}"
    d = [first_pay + pd.DateOffset(months=m - 1) for m in w]
    return f"{d[0]:%m/%y}-{d[1]:%m/%y}"


def yt_table(deal: Deal, scenarios: dict[str, Assumptions], name: str, price: float,
             curve: Curve, index=0.0, first_pay: pd.Timestamp | None = None) -> pd.DataFrame:
    """Bloomberg-YT-style grid: one column per prepay scenario, at a fixed price."""
    face = next(t.balance for t in deal.tranches if t.name == name)
    cols = {}
    for label, a in scenarios.items():
        res = run_waterfall(deal, a, index)
        m = metrics(res.cash_flow(name), res.principal(name), price, face, curve)
        cols[label] = {"Price": f"{price:.3f}", "Yield (%)": f"{100 * m['yield']:.3f}",
                       "Avg Life (yr)": f"{m['wal']:.2f}", "Mod Dur": f"{m['mod_dur']:.2f}",
                       "Principal Window": _fmt_window(m["window"], first_pay),
                       "I-Spread (bp)": f"{1e4 * m['i_spread']:.0f}"}
    return pd.DataFrame(cols)


def stress_grid(deal: Deal, base: Assumptions, prices: dict[str, float],
                prepay_mults, loss_mults, index=0.0) -> pd.DataFrame:
    """Yield, duration and duration-adjusted yield of each class at its base-case
    price, over a prepay-stress x loss-stress grid (long format)."""
    rows = []
    for pm in prepay_mults:
        for lm in loss_mults:
            res = run_waterfall(deal, base.stressed(pm, lm), index)
            for t in deal.tranches:
                cf = res.cash_flow(t.name)
                y = yield_from_price(cf, prices[t.name], t.balance)
                d = modified_duration(cf, y)
                rows.append({"class": t.name, "prepay_mult": pm, "loss_mult": lm, "yield": y,
                             "mod_dur": d, "dur_adj_yield": y / d,
                             "principal_loss": res.classes[t.name]["writedown"].sum() / t.balance,
                             "cnl": res.deal["cum_net_loss"].iloc[-1]})
    return pd.DataFrame(rows)


def breakeven_loss_mult(deal: Deal, base: Assumptions, name: str, hi: float = 100.0,
                        tol: float = 0.01, index=0.0) -> tuple[float, float]:
    """Smallest loss multiple at which the class loses principal, and the pool's
    cumulative net loss at that multiple (bisection)."""
    face = next(t.balance for t in deal.tranches if t.name == name)
    loses = lambda m: run_waterfall(deal, base.stressed(1, m), index).classes[name][
        "writedown"].sum() > 1e-6 * face
    lo = 0.0
    if not loses(hi):
        return np.inf, np.nan
    while hi - lo > tol:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if loses(mid) else (mid, hi)
    return hi, float(run_waterfall(deal, base.stressed(1, hi), index).deal["cum_net_loss"].iloc[-1])


def equity_yield(res: WaterfallResult, deal: Deal) -> float:
    """IRR of the residual for an investor who funds the initial OC and reserve at par."""
    invested = deal.pool.total - sum(t.balance for t in deal.tranches) + \
        (deal.reserve * deal.original if deal.reserve_balance is None else deal.reserve_balance)
    return yield_from_price(res.cash_flow("Equity"), 100.0, invested)
