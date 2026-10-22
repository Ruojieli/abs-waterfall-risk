"""Collateral and waterfall: closed-form answers, conservation laws, and monotonicity."""
from dataclasses import replace

import numpy as np
import pytest

from abswf.analytics import wal
from abswf.collateral import (Assumptions, Pool, abs_to_smm, cpr_to_smm, project_pool, ramp,
                              smm_to_abs, smm_to_cpr)
from abswf.deals import SYNTH_BASE, synthetic_deal
from abswf.waterfall import Tranche, run_waterfall

NO_LOSS = Assumptions()


def one_line(rate=0.06, term=60, floating=False, balance=1e6):
    return Pool.lines([{"balance": balance, "rate": rate, "term": term, "age": 0,
                        "floating": floating}])


def test_speed_conversions():
    assert cpr_to_smm(0.06) == pytest.approx(0.0051430, abs=1e-7)
    assert smm_to_cpr(cpr_to_smm(0.237)) == pytest.approx(0.237)
    assert abs_to_smm(0.015, 1) == pytest.approx(0.015)
    assert abs_to_smm(0.015, 11) == pytest.approx(0.015 / 0.85)
    assert smm_to_abs(abs_to_smm(0.013, 25), 25) == pytest.approx(0.013)
    assert ramp(0.01, 0.06, 6, 8).tolist() == pytest.approx([.01, .02, .03, .04, .05, .06, .06, .06])


def test_zero_default_zero_prepay_is_level_pay_annuity():
    P, r, n = 1e6, 0.06 / 12, 60
    cf = project_pool(one_line(), NO_LOSS, cleanup_call=None)
    pmt = P * r / (1 - (1 + r) ** -n)
    k = np.arange(1, n + 1)
    balance = P * ((1 + r) ** n - (1 + r) ** k) / ((1 + r) ** n - 1)
    assert len(cf) == n
    np.testing.assert_allclose(cf["interest"] + cf["sched_principal"], pmt, rtol=1e-10)
    np.testing.assert_allclose(cf["end_balance"], balance, atol=1e-6)
    assert cf["prepay"].sum() == 0 and cf["loss"].sum() == 0


def test_floating_line_with_flat_index_matches_fixed_line():
    fixed = project_pool(one_line(rate=0.06), SYNTH_BASE, cleanup_call=None)
    floating = project_pool(one_line(rate=0.02, floating=True), SYNTH_BASE, index=0.04,
                            cleanup_call=None)
    np.testing.assert_allclose(fixed.to_numpy(), floating.to_numpy(), rtol=1e-12)


@pytest.mark.parametrize("lag", [0, 4])
def test_pool_balance_identity_and_cleanup_call(lag):
    deal = synthetic_deal()
    a = replace(SYNTH_BASE.stressed(1.0, 5.0), recovery_lag=lag)
    cf = project_pool(deal.pool, a, cleanup_call=0.10)
    out = cf[["sched_principal", "prepay", "recovery", "loss", "call_principal"]].sum(axis=1)
    np.testing.assert_allclose(cf["begin_balance"] - out, cf["end_balance"], atol=1e-4)
    assert out.sum() == pytest.approx(deal.pool.total)
    # called in the first month the pool factor is at or below 10%
    uncalled = project_pool(deal.pool, a, cleanup_call=None)["end_balance"]
    t = cf.index[-1]
    assert uncalled[t] <= 0.10 * deal.pool.total < uncalled[t - 1]
    assert cf["call_principal"].iloc[-1] > 0 and cf["end_balance"].iloc[-1] == 0


def pro_rata_deal():
    return synthetic_deal(pro_rata=True, cnl_trigger=0.01)


def floating_deal():
    tr = (Tranche("A1", 430e6, 0.005, floating=True, group="A"), Tranche("A2", 430e6, 0.053, group="A"),
          Tranche("B", 45e6, 0.0575), Tranche("C", 35e6, 0.0605), Tranche("D", 25e6, 0.0670))
    return replace(synthetic_deal(), tranches=tr)


CASES = [(synthetic_deal(), SYNTH_BASE), (synthetic_deal(), SYNTH_BASE.stressed(0.5, 12)),
         (synthetic_deal(cleanup_call=None), SYNTH_BASE.stressed(2.0, 20)),
         (pro_rata_deal(), SYNTH_BASE.stressed(1.0, 3)), (floating_deal(), SYNTH_BASE.stressed(1, 8))]


@pytest.mark.parametrize("deal,a", CASES)
def test_cash_and_par_conservation(deal, a):
    res = run_waterfall(deal, a, index=0.04)
    d, c = res.deal, res.classes
    paid = sum(c[n]["interest"] + c[n]["principal"] for n in deal.names)
    sources = res.collateral["collections"] + d["reserve_draw"]
    uses = d["fee"] + paid + d["reserve_deposit"] + d["residual"]
    np.testing.assert_allclose(sources, uses, atol=1e-5)          # every period
    for t in deal.tranches:                                        # every dollar of par
        x = c[t.name]
        assert x["principal"].sum() + x["writedown"].sum() + x["end_balance"].iloc[-1] == \
            pytest.approx(t.balance)
        assert x["end_balance"].iloc[-1] == pytest.approx(0, abs=1e-6)
    assert (d["oc"] >= -1e-6).all() and (d["residual"] >= -1e-9).all()


def test_losses_are_allocated_bottom_up():
    deal = synthetic_deal()
    for mult in (8, 10, 12, 15, 20):
        wd = {n: run_waterfall(deal, SYNTH_BASE.stressed(1, mult)).classes[n]["writedown"].sum()
              for n in deal.names}
        for junior, senior in zip(deal.names[::-1], deal.names[-2::-1]):
            if wd[senior] > 0:      # a senior class loses only after the class below is gone
                face = next(t.balance for t in deal.tranches if t.name == junior)
                assert wd[junior] == pytest.approx(face)


def test_higher_cdr_never_increases_subordinate_cash():
    deal = synthetic_deal()
    mults = [0, 0.5, 1, 2, 3, 4, 6, 8, 10, 12, 15, 20]
    for name in ["B", "C", "D", "Equity"]:
        total = [run_waterfall(deal, SYNTH_BASE.stressed(1, m)).cash_flow(name).sum() for m in mults]
        assert np.all(np.diff(total) <= 1e-6), name


def test_senior_wal_shortens_as_cpr_rises():
    deal = synthetic_deal()
    wals = [wal(run_waterfall(deal, replace(SYNTH_BASE, prepay=c, prepay_type="CPR")).principal("A"))
            for c in np.arange(0, 0.45, 0.05)]
    assert np.all(np.diff(wals) < 0)


def test_trigger_switches_pro_rata_to_sequential():
    deal = pro_rata_deal()
    res = run_waterfall(deal, SYNTH_BASE.stressed(1, 3))
    seq = res.deal["sequential"].to_numpy()
    assert not seq[0] and seq[-1]                                  # trigger tripped mid-life
    a, b = res.classes["A"], res.classes["B"]
    before = ~seq & (a["principal"] > 0).to_numpy()
    np.testing.assert_allclose((b["principal"] / a["principal"])[before],
                               (b["begin_balance"] / a["begin_balance"])[before], rtol=1e-9)
    after = seq & (a["end_balance"] > 1).to_numpy()            # A still outstanding
    assert after.sum() > 5 and (b["principal"][after] == 0).all()
