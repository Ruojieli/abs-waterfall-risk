"""Market risk for ABS tranches: PV01 and CS01 by bump-and-reprice, static stress
scenarios, and a historical-simulation VaR from daily rate and spread changes.

Each class is priced off the Treasury zero curve plus its own Z-spread, solved from
its base-case price. PV01 bumps the par curve (and any floating-rate index) by 1bp
with the Z-spread fixed; CS01 bumps the Z-spread. Both are the price gain for a 1bp
fall, per 100 face. Collateral cash flows do not depend on rates here (auto-loan
prepayments are treated as rate-insensitive), so fixed-rate classes have PV01 close
to CS01 and floating-rate classes have PV01 near zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from .analytics import (modified_duration, price_on_curve, yield_from_price, z_spread)
from .collateral import Assumptions
from .curve import Curve
from .waterfall import Deal, run_waterfall


def _prices(deal: Deal, a: Assumptions, curve: Curve, z: dict, index: float) -> dict:
    res = run_waterfall(deal, a, index)
    return {t.name: price_on_curve(res.cash_flow(t.name), t.balance, curve, z[t.name])
            for t in deal.tranches}


def base_z_spreads(deal: Deal, a: Assumptions, prices: dict, curve: Curve, index=0.0) -> dict:
    res = run_waterfall(deal, a, index)
    return {t.name: z_spread(res.cash_flow(t.name), prices[t.name], t.balance, curve)
            for t in deal.tranches}


def sensitivities(deal: Deal, a: Assumptions, prices: dict, curve: Curve,
                  index: float = 0.0, bp: float = 1.0) -> pd.DataFrame:
    z = base_z_spreads(deal, a, prices, curve, index)
    h = bp / 1e4
    up = _prices(deal, a, curve.shifted(bp), z, index + h)
    dn = _prices(deal, a, curve.shifted(-bp), z, index - h)
    zu = _prices(deal, a, curve, {k: v + h for k, v in z.items()}, index)
    zd = _prices(deal, a, curve, {k: v - h for k, v in z.items()}, index)
    res = run_waterfall(deal, a, index)
    rows = []
    for t in deal.tranches:
        cf = res.cash_flow(t.name)
        y = yield_from_price(cf, prices[t.name], t.balance)
        d = modified_duration(cf, y)
        rows.append({"class": t.name, "price": prices[t.name], "yield": y, "z_spread": z[t.name],
                     "mod_dur": d, "pv01": (dn[t.name] - up[t.name]) / 2 / bp,
                     "cs01": (zd[t.name] - zu[t.name]) / 2 / bp,
                     "dur_x_price": d * prices[t.name] * 1e-4})
    return pd.DataFrame(rows).set_index("class")


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str                                  # "historical" or "hypothetical"
    rates_bp: float = 0.0                      # parallel shift of the par curve
    curve_bp: pd.Series | None = None          # or a shift by tenor (index = years)
    spread_bp: dict = field(default_factory=dict)   # by class; "*" applies to all
    prepay_mult: float = 1.0
    loss_mult: float = 1.0
    severity: float | None = None

    def apply(self, curve: Curve, a: Assumptions) -> tuple[Curve, Assumptions]:
        shift = np.full(len(curve.tenors), self.rates_bp)
        if self.curve_bp is not None:
            shift = shift + np.interp(curve.tenors, self.curve_bp.index, self.curve_bp.values)
        a2 = a.stressed(self.prepay_mult, self.loss_mult)
        if self.severity is not None:
            a2 = replace(a2, severity=self.severity)
        return curve.shifted(by_tenor=shift / 1e4), a2

    def spread(self, name: str) -> float:
        return self.spread_bp.get(name, self.spread_bp.get("*", 0.0)) / 1e4


def run_stress(deal: Deal, a: Assumptions, prices: dict, curve: Curve, positions: dict,
               scenarios: list[Scenario], index: float = 0.0) -> pd.DataFrame:
    """Full revaluation of each class under each scenario; P&L on the face held."""
    z = base_z_spreads(deal, a, prices, curve, index)
    rows = []
    for s in scenarios:
        c2, a2 = s.apply(curve, a)
        short_end = float(c2.par_yield(1 / 12) - curve.par_yield(1 / 12))
        p = _prices(deal, a2, c2, {k: v + s.spread(k) for k, v in z.items()}, index + short_end)
        row = {"scenario": s.name, "kind": s.kind}
        for k in positions:
            row[f"{k} price chg"] = p[k] - prices[k]
        row["P&L"] = sum(positions[k] * (p[k] - prices[k]) / 100 for k in positions)
        rows.append(row)
    return pd.DataFrame(rows).set_index("scenario")


def _tail(pnl: np.ndarray, alpha: float) -> tuple[float, float]:
    var = -np.quantile(pnl, 1 - alpha)
    return float(var), float(-pnl[pnl <= -var].mean())


def historical_var(deal: Deal, a: Assumptions, prices: dict, curve: Curve, positions: dict,
                   rate_changes: pd.DataFrame, spread_changes: pd.DataFrame | None,
                   alpha: float = 0.99) -> dict:
    """1-day historical-simulation VaR and ES with full revaluation.

    rate_changes: daily par-yield changes (decimal), columns = tenor in years.
    spread_changes: daily spread changes (decimal), columns = class names.
    Cash flows are held at the base case (fixed-rate classes only).
    """
    res = run_waterfall(deal, a)
    cfs = {k: res.cash_flow(k) for k in positions}
    face = {t.name: t.balance for t in deal.tranches}
    z = {k: z_spread(cfs[k], prices[k], face[k], curve) for k in positions}
    tenors = rate_changes.columns.to_numpy(float)

    def pnl(dr: np.ndarray | None, ds: pd.Series | None) -> float:
        c = curve if dr is None else curve.shifted(by_tenor=np.interp(curve.tenors, tenors, dr))
        return sum(positions[k] / 100 * (price_on_curve(cfs[k], face[k], c,
                   z[k] + (0.0 if ds is None else ds[k])) - prices[k]) for k in positions)

    days = rate_changes.index if spread_changes is None else \
        rate_changes.index.intersection(spread_changes.index)
    out = {"days": len(days), "start": days[0], "end": days[-1]}
    legs = {"Rates only": (True, False)}
    if spread_changes is not None:
        legs = {"Rates + spreads": (True, True), **legs, "Spreads only": (False, True)}
    for label, (use_r, use_s) in legs.items():
        p = np.array([pnl(rate_changes.loc[d].to_numpy() if use_r else None,
                          spread_changes.loc[d] if use_s else None) for d in days])
        var, es = _tail(p, alpha)
        out[label] = {"VaR": var, "ES": es, "pnl": p}
    return out
