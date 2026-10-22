"""Monthly liability waterfall: sequential-pay notes plus a residual (equity) class.

Each month, available funds (pool collections) pay, in order:
  1. the servicing fee
  2. note interest, top-down (shortfalls carry forward; the reserve covers them)
  3. note principal, top-down: the principal distribution amount brings the notes
     down to (pool balance - target OC), so excess spread first covers losses and
     builds overcollateralization before anything reaches equity
  4. the reserve account top-up to its required balance
  5. the residual, to equity
Losses are allocated bottom-up: if the notes exceed the pool after payments, the
most junior note is written down first (equity's OC cushion is already gone).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .collateral import Assumptions, Pool, _vector, project_pool


@dataclass(frozen=True)
class Tranche:
    name: str
    balance: float
    coupon: float              # fixed coupon, or the margin over the index if floating
    floating: bool = False
    group: str | None = None   # classes sharing a group split principal pro rata


@dataclass(frozen=True)
class Deal:
    name: str
    pool: Pool
    tranches: tuple[Tranche, ...]        # most senior first
    servicing_fee: float = 0.01
    oc_target: float = 0.0               # target OC, fraction of the current pool
    oc_floor: float = 0.0                # floor on target OC, fraction of the original pool
    oc_fixed: float = 0.0                # dollar amount added to target OC (YSOC-style deals)
    reserve: float = 0.0                 # required reserve, fraction of the original pool
    reserve_balance: float | None = None  # starting reserve (default: fully funded)
    pro_rata: bool = False               # pay principal pro rata until the trigger trips
    cnl_trigger: float | None = None     # cumulative net loss / original pool -> sequential
    cleanup_call: float | None = 0.10
    original_balance: float | None = None  # cutoff pool, if the deal is already seasoned

    @property
    def original(self) -> float:
        return self.original_balance or self.pool.total

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.tranches]


@dataclass
class WaterfallResult:
    collateral: pd.DataFrame
    deal: pd.DataFrame
    classes: dict[str, pd.DataFrame]

    def cash_flow(self, name: str) -> np.ndarray:
        if name == "Equity":
            return self.deal["residual"].to_numpy()
        c = self.classes[name]
        return (c["interest"] + c["principal"]).to_numpy()

    def principal(self, name: str) -> np.ndarray:
        return self.classes[name]["principal"].to_numpy()


def _allocate(amount: float, bal: np.ndarray, groups: list[str], sequential: bool) -> np.ndarray:
    paid = np.zeros_like(bal)
    if amount <= 0 or bal.sum() <= 0:
        return paid
    if not sequential:
        return amount * bal / bal.sum()
    for g in dict.fromkeys(groups):          # groups in order of seniority
        m = np.array([x == g for x in groups])
        need = bal[m].sum()
        x = min(amount, need)
        if need > 0:
            paid[m] = x * bal[m] / need
        amount -= x
        if amount <= 1e-9:
            break
    return paid


def run_waterfall(deal: Deal, a: Assumptions, index: np.ndarray | float = 0.0) -> WaterfallResult:
    cf = project_pool(deal.pool, a, index, deal.cleanup_call, deal.original)
    idx = _vector(index, len(cf))
    tr = deal.tranches
    groups = [t.group or t.name for t in tr]
    bal = np.array([t.balance for t in tr], float)
    short = np.zeros(len(tr))
    reserve_req = deal.reserve * deal.original
    R = reserve_req if deal.reserve_balance is None else deal.reserve_balance
    sequential, cum_loss = not deal.pro_rata, 0.0
    rec = {t.name: [] for t in tr}
    rows = []

    for i, (month, c) in enumerate(cf.iterrows()):
        af, beg, draw = c["collections"], bal.copy(), 0.0
        fee = min(af, deal.servicing_fee / 12 * c["begin_balance"])
        af -= fee

        rates = np.array([t.coupon + (idx[i] if t.floating else 0.0) for t in tr])
        due = beg * rates / 12 + short
        interest = np.zeros(len(tr))
        for k in range(len(tr)):
            pay = min(due[k], af)
            af -= pay
            if pay < due[k] and R > 0:       # reserve covers interest shortfalls
                x = min(R, due[k] - pay)
                R, draw, pay = R - x, draw + x, pay + x
            interest[k] = pay
        short = due - interest

        pool_end = c["end_balance"]
        cum_loss += c["loss"]
        if deal.cnl_trigger is not None and cum_loss / deal.original > deal.cnl_trigger:
            sequential = True                # trigger breach is permanent
        target_oc = min(pool_end, max(deal.oc_target * pool_end, deal.oc_floor * deal.original)
                        + deal.oc_fixed)
        pda = max(0.0, beg.sum() - (pool_end - target_oc))
        pay = min(pda, af)
        af -= pay
        must = max(0.0, beg.sum() - pool_end)  # keeps the notes covered by the pool
        if pay < must and R > 0:
            x = min(R, must - pay)
            R, draw, pay = R - x, draw + x, pay + x
        principal = _allocate(pay, beg, groups, sequential)
        bal = beg - principal

        req = min(reserve_req, bal.sum())
        deposit = min(af, max(0.0, req - R))
        af -= deposit
        R += deposit
        if R > req:                          # release the excess reserve to equity
            draw, af, R = draw + R - req, af + R - req, req

        writedown = np.zeros(len(tr))
        excess = bal.sum() - pool_end
        for k in reversed(range(len(tr))):
            if excess <= 1e-9:
                break
            writedown[k] = min(bal[k], excess)
            excess -= writedown[k]
        bal = bal - writedown
        short = np.where(bal > 1e-9, short, 0.0)   # a written-off class is extinguished

        for k, t in enumerate(tr):
            rec[t.name].append({"month": month, "begin_balance": beg[k], "interest_due": due[k],
                                "interest": interest[k], "principal": principal[k],
                                "writedown": writedown[k], "end_balance": bal[k],
                                "shortfall": short[k]})
        rows.append({"month": month, "available": c["collections"], "fee": fee,
                     "reserve_draw": draw, "reserve_deposit": deposit, "reserve_balance": R,
                     "residual": af, "notes_end": bal.sum(), "pool_end": pool_end,
                     "oc": pool_end - bal.sum(), "cum_net_loss": cum_loss / deal.original,
                     "sequential": sequential})

    classes = {n: pd.DataFrame(v).set_index("month") for n, v in rec.items()}
    return WaterfallResult(cf, pd.DataFrame(rows).set_index("month"), classes)
