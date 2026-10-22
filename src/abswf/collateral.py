"""Collateral cash flows for an amortizing loan pool, projected by rep line.

Speed conventions (all rates are decimals):
  SMM  single monthly mortality (voluntary prepayment), CPR = 1 - (1 - SMM)^12
  MDR  monthly default rate,                            CDR = 1 - (1 - MDR)^12
  ABS  auto-ABS speed: prepayments each month as a fraction of the ORIGINAL number
       of loans, so SMM_t = ABS / (1 - ABS * (age_t - 1)) with age_t the loan age.

Each month, defaults hit the beginning performing balance first, the survivors pay
their scheduled (level-pay) principal, and the remainder prepays at the SMM. Defaulted
balances stop paying interest and are liquidated `recovery_lag` months later, when the
recovery (1 - severity) arrives as cash and the rest is realized as a loss.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


def cpr_to_smm(cpr):
    return 1 - (1 - np.asarray(cpr, float)) ** (1 / 12)


def smm_to_cpr(smm):
    return 1 - (1 - np.asarray(smm, float)) ** 12


cdr_to_mdr, mdr_to_cdr = cpr_to_smm, smm_to_cpr


def abs_to_smm(abs_speed, age):
    """SMM implied by an ABS speed for loans of the given age (1 = first payment)."""
    denom = 1 - np.asarray(abs_speed, float) * (np.asarray(age, float) - 1)
    return np.where(denom > abs_speed, abs_speed / np.maximum(denom, 1e-12), 1.0)


def smm_to_abs(smm, age):
    """Inverse of abs_to_smm."""
    smm = np.asarray(smm, float)
    return smm / (1 + smm * (np.asarray(age, float) - 1))


def ramp(start: float, plateau: float, ramp_months: int, n: int = 120) -> np.ndarray:
    """Speed vector rising linearly from `start` in month 1 to `plateau` in month
    `ramp_months`, flat afterwards (the usual base-case shape for CDR and CPR)."""
    t = np.arange(1, n + 1)
    if ramp_months <= 1:
        return np.full(n, float(plateau))
    v = start + (plateau - start) * (t - 1) / (ramp_months - 1)
    return np.where(t >= ramp_months, plateau, v)


def _vector(x, n: int) -> np.ndarray:
    """Scalar or vector of monthly values, padded with its last value to length n."""
    v = np.atleast_1d(np.asarray(x, float))
    return np.concatenate([v, np.full(max(n - len(v), 0), v[-1])])[:n]


@dataclass(frozen=True)
class Assumptions:
    """Collateral performance assumptions; prepay and cdr may be monthly vectors."""
    prepay: float | np.ndarray = 0.0     # annual CPR, or ABS speed if prepay_type="ABS"
    prepay_type: str = "CPR"
    cdr: float | np.ndarray = 0.0        # annual default rate
    severity: float = 0.50               # loss given default
    recovery_lag: int = 0                # months from default to liquidation
    prepay_mult: float = 1.0             # multiplicative stress on the prepay speed
    loss_mult: float = 1.0               # multiplicative stress on the default rate

    def stressed(self, prepay_mult: float = 1.0, loss_mult: float = 1.0) -> "Assumptions":
        return replace(self, prepay_mult=self.prepay_mult * prepay_mult,
                       loss_mult=self.loss_mult * loss_mult)

    def smm(self, n: int, age0: np.ndarray) -> np.ndarray:
        """Monthly SMM, shape (n months, n lines)."""
        speed = _vector(self.prepay, n) * self.prepay_mult
        if self.prepay_type.upper() == "ABS":
            age = np.asarray(age0, float)[None, :] + np.arange(1, n + 1)[:, None]
            return abs_to_smm(speed[:, None], age)
        smm = cpr_to_smm(np.minimum(speed, 1.0))
        return np.repeat(smm[:, None], len(np.atleast_1d(age0)), axis=1)

    def mdr(self, n: int) -> np.ndarray:
        return cdr_to_mdr(np.minimum(_vector(self.cdr, n) * self.loss_mult, 1.0))


@dataclass(frozen=True)
class Pool:
    """Rep lines of a loan pool. A loan-level pool is just one line per loan."""
    balance: np.ndarray
    rate: np.ndarray           # gross APR; for floating lines, the margin over the index
    term: np.ndarray           # remaining term in months
    age: np.ndarray            # months since origination
    floating: np.ndarray       # bool per line

    @classmethod
    def lines(cls, rows: list[dict]) -> "Pool":
        df = pd.DataFrame(rows)
        return cls(df["balance"].to_numpy(float), df["rate"].to_numpy(float),
                   df["term"].to_numpy(int), df.get("age", pd.Series(0, df.index)).to_numpy(int),
                   df.get("floating", pd.Series(False, df.index)).to_numpy(bool))

    @property
    def total(self) -> float:
        return float(self.balance.sum())

    @property
    def wac(self) -> float:
        return float(self.rate @ self.balance / self.total)


def level_payment(balance, rate_m, n):
    """Level monthly payment that amortizes `balance` over n months at monthly rate."""
    balance, rate_m, n = (np.asarray(a, float) for a in (balance, rate_m, n))
    n = np.maximum(n, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = np.where(rate_m > 0, rate_m / (1 - (1 + rate_m) ** -n), 1 / n)
    return balance * annuity


def project_pool(pool: Pool, a: Assumptions, index: np.ndarray | float = 0.0,
                 cleanup_call: float | None = 0.10,
                 original_balance: float | None = None) -> pd.DataFrame:
    """Monthly pool cash flows (rows = payment months 1..T).

    `index` is the annual floating-rate index path for floating lines. If
    `cleanup_call` is set, the pool is bought out on the first month its balance
    (performing plus defaulted-but-unliquidated) falls to that fraction of the
    original balance (default: today's balance): performing loans at par, pending
    recoveries at their expected value.
    """
    lag = int(a.recovery_lag)
    T = int(pool.term.max()) + lag
    smm, mdr = a.smm(T, pool.age), a.mdr(T)
    idx = _vector(index, T)
    perf = pool.balance.astype(float).copy()
    pending = np.zeros(T + lag + 1)              # defaulted balance by liquidation month
    rows = []
    for t in range(1, T + 1):
        rate_m = (pool.rate + np.where(pool.floating, idx[t - 1], 0.0)) / 12
        n_left = pool.term - (t - 1)
        beg = perf.sum() + pending[t:].sum()
        default = perf * mdr[t - 1]
        live = perf - default
        interest = live * rate_m
        sched = np.where(n_left > 1, level_payment(live, rate_m, n_left) - interest, live)
        sched = np.clip(sched, 0, live)
        prepay = (live - sched) * smm[t - 1]
        perf = live - sched - prepay
        pending[t + lag] += default.sum()
        liquidated = pending[t]
        rows.append({"month": t, "begin_balance": beg, "interest": interest.sum(),
                     "sched_principal": sched.sum(), "prepay": prepay.sum(),
                     "default": default.sum(), "recovery": liquidated * (1 - a.severity),
                     "loss": liquidated * a.severity, "call_principal": 0.0,
                     "performing": perf.sum(), "pipeline": pending[t + 1:].sum()})
    cf = pd.DataFrame(rows).set_index("month")
    cf["end_balance"] = cf["performing"] + cf["pipeline"]

    if cleanup_call is not None:
        threshold = cleanup_call * (original_balance or pool.total)
        hit = np.flatnonzero(cf["end_balance"].to_numpy() <= threshold)
        if len(hit):
            t = int(cf.index[hit[0]])
            cf = cf.loc[:t].copy()
            pipe = cf.at[t, "pipeline"]
            cf.at[t, "call_principal"] = cf.at[t, "performing"]
            cf.at[t, "recovery"] += pipe * (1 - a.severity)
            cf.at[t, "loss"] += pipe * a.severity
            cf.loc[t, ["performing", "pipeline", "end_balance"]] = 0.0
    cf["collections"] = cf[["interest", "sched_principal", "prepay", "recovery",
                            "call_principal"]].sum(axis=1)
    return cf


def cumulative_net_loss(cf: pd.DataFrame, pool: Pool) -> float:
    return float(cf["loss"].sum() / pool.total)
