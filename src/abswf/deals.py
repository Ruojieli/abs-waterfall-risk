"""The two deals in the reports.

SYNTH 2026-1 is a synthetic prime-auto deal (sequential A/B/C/D plus equity). It is
not modeled on any issuer. FORDO 2026-A is built from public SEC filings: the pool
from the ABS-EE loan tape, the note balances and coupons from the 10-D investor
report, with a simplified waterfall (see README).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .collateral import Assumptions, Pool, ramp
from .waterfall import Deal, Tranche

SYNTH_BASE = Assumptions(prepay=0.014, prepay_type="ABS", cdr=ramp(0.004, 0.016, 12),
                         severity=0.50, recovery_lag=4)
# Assumed new-issue spreads over the interpolated Treasury at WAL (not market quotes)
SYNTH_SPREADS = {"A": 0.0055, "B": 0.0085, "C": 0.0115, "D": 0.0175}


def synthetic_deal(**overrides) -> Deal:
    pool = Pool.lines([{"balance": 300e6, "rate": 0.0750, "term": 48, "age": 8},
                       {"balance": 450e6, "rate": 0.0825, "term": 60, "age": 6},
                       {"balance": 250e6, "rate": 0.0900, "term": 72, "age": 4}])
    tranches = (Tranche("A", 860e6, 0.0530), Tranche("B", 45e6, 0.0575),
                Tranche("C", 35e6, 0.0605), Tranche("D", 25e6, 0.0670))
    kw = dict(servicing_fee=0.01, oc_target=0.05, oc_floor=0.01, reserve=0.005,
              cleanup_call=0.10)
    kw.update(overrides)
    return Deal("SYNTH 2026-1", pool, tranches, **kw)


FORDO_ORIGINAL_POOL = 1_488_961_023.25    # initial pool balance, 10-D section I
FORDO_FLOAT_MARGIN = 0.0043               # Class A-2b: 30-day average SOFR + 0.43%


def fordo_deal(lines: pd.DataFrame, report: pd.Series) -> Deal:
    """FORDO 2026-A after the latest payment date, from the rep lines and 10-D row.

    Classes A-2a/A-2b share principal pro rata (group "A-2"); A-2b floats. In the
    10-D, target OC = yield supplement OC (YSOC) + a constant dollar amount; the model
    keeps that constant and holds YSOC at its current share of the pool.
    """
    pool = Pool.lines(lines.to_dict("records"))
    notes, rates = json.loads(report["notes_end"]), json.loads(report["note_rates"])
    tranches = []
    for cls, bal in notes.items():
        if bal <= 0:
            continue
        name = cls.replace("Class ", "")
        fl = name == "A-2b"
        tranches.append(Tranche(name, bal, FORDO_FLOAT_MARGIN if fl else rates[cls], floating=fl,
                                group="A-2" if name.startswith("A-2") else None))
    return Deal("FORDO 2026-A", pool, tuple(tranches), servicing_fee=0.01,
                oc_target=report["ysoc"] / report["pool_end"],
                oc_fixed=report["target_oc"] - report["ysoc"],
                reserve=report["reserve_end"] / FORDO_ORIGINAL_POOL,
                reserve_balance=report["reserve_end"], cleanup_call=0.10,
                original_balance=FORDO_ORIGINAL_POOL)


def fordo_index(report: pd.Series) -> float:
    """30-day average SOFR implied by the A-2b coupon in the latest 10-D."""
    return json.loads(report["note_rates"])["Class A-2b"] - FORDO_FLOAT_MARGIN


def calibrated_assumptions(stats: pd.DataFrame, months: int | None = 4,
                           recovery_lag: int = 3) -> Assumptions:
    """Flat ABS speed and CDR at their balance-weighted realized averages over the
    trailing `months` (all months if None), and severity from cumulative net losses
    over cumulative gross defaults since cutoff."""
    recent = stats if months is None else stats.iloc[-months:]
    w = recent["begin_balance"] / recent["begin_balance"].sum()
    sev = (stats["chargeoff"].sum() - stats["recoveries"].sum()) / stats["gross_default"].sum()
    return Assumptions(prepay=float(w @ recent["abs"]), prepay_type="ABS",
                       cdr=float(w @ recent["cdr"]), severity=float(np.clip(sev, 0, 1)),
                       recovery_lag=recovery_lag)
