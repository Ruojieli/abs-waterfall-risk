"""Analytics, curve, risk and EDGAR parsing checked against closed-form answers."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from abswf.analytics import (convexity, macaulay, modified_duration, price_from_yield,
                             price_on_curve, yield_from_price)
from abswf.curve import Curve
from abswf.deals import SYNTH_BASE, synthetic_deal
from abswf.edgar import parse_tape, rep_lines, tape_stats
from abswf.risk import historical_var, sensitivities
from abswf.waterfall import Tranche, run_waterfall

# A sloped curve shaped like late-2026 Treasuries (par yields, %)
SLOPED = Curve(np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]),
               np.array([4.09, 4.25, 4.35, 4.52, 4.81, 4.89, 4.95, 5.03, 5.13, 5.47, 5.42]) / 100)


def base_prices(deal, a, y=0.055):
    res = run_waterfall(deal, a)
    return {t.name: price_from_yield(res.cash_flow(t.name), y, t.balance) for t in deal.tranches}


def test_price_yield_round_trip_and_floor():
    cf = np.r_[np.full(23, 0.5), 100.5]
    assert yield_from_price(cf, price_from_yield(cf, 0.0537, 100), 100) == pytest.approx(0.0537)
    assert price_from_yield(np.r_[np.zeros(5), 3.0, np.zeros(5), 103.0], 0.06, 100) == \
        pytest.approx(100.0, abs=1e-12)                              # semiannual par bond
    assert yield_from_price(np.full(12, 0.1), 100, 100) == -1.0      # wiped out: floored


def test_zero_coupon_duration_and_convexity():
    cf, y, h = np.r_[np.zeros(23), 100.0], 0.05, 1e-5
    assert macaulay(cf, y) == pytest.approx(2.0)
    assert modified_duration(cf, y) == pytest.approx(2.0 / 1.025)
    p = lambda yy: price_from_yield(cf, yy, 100)
    fd = (p(y + h) - 2 * p(y) + p(y - h)) / (h * h) / p(y)
    assert convexity(cf, y) == pytest.approx(fd, rel=1e-4)


def test_flat_curve_bootstraps_to_flat_zeros_and_prices_par_bonds():
    c = Curve.flat(0.05)
    np.testing.assert_allclose(c.zero(np.arange(1, 61) / 2), 0.05, atol=1e-12)
    bond = np.r_[np.tile(np.r_[np.zeros(5), 2.5], 19), np.zeros(5), 102.5]   # 10y semiannual
    assert price_on_curve(bond, 100, c, 0.0) == pytest.approx(100.0, abs=1e-9)


def test_sloped_curve_reprices_its_par_bonds():
    for T in (2, 5, 10):
        c = float(SLOPED.par_yield(T))
        bond = np.r_[np.tile(np.r_[np.zeros(5), 50 * c], 2 * T - 1), np.zeros(5), 100 + 50 * c]
        assert price_on_curve(bond, 100, SLOPED, 0.0) == pytest.approx(100.0, abs=1e-6)


@pytest.mark.parametrize("curve,tol", [(Curve.flat(0.05), 0.003), (SLOPED, 0.02)])
def test_pv01_matches_modified_duration_times_price(curve, tol):
    deal = synthetic_deal()
    s = sensitivities(deal, SYNTH_BASE, base_prices(deal, SYNTH_BASE), curve)
    np.testing.assert_allclose(s["pv01"], s["dur_x_price"], rtol=tol)
    np.testing.assert_allclose(s["pv01"], s["cs01"], rtol=tol)   # fixed rate: rates ~ spreads


def test_floater_has_spread_risk_but_little_rate_risk():
    tr = (Tranche("A", 860e6, 0.005, floating=True),) + synthetic_deal().tranches[1:]
    deal = replace(synthetic_deal(), tranches=tr)
    res = run_waterfall(deal, SYNTH_BASE, 0.045)
    prices = {t.name: price_from_yield(res.cash_flow(t.name), 0.055, t.balance) for t in tr}
    s = sensitivities(deal, SYNTH_BASE, prices, SLOPED, index=0.045)
    assert abs(s.loc["A", "pv01"]) < 0.1 * s.loc["A", "cs01"]


def test_var_scenario_pnl_matches_pv01_for_a_parallel_move():
    deal = synthetic_deal()
    prices = base_prices(deal, SYNTH_BASE)
    pos = {"A": 10e6, "D": 5e6}
    moves = pd.DataFrame([np.full(len(SLOPED.tenors), 1e-4)], columns=SLOPED.tenors,
                         index=pd.to_datetime(["2026-01-02"]))
    out = historical_var(deal, SYNTH_BASE, prices, SLOPED, pos, moves, None)
    s = sensitivities(deal, SYNTH_BASE, prices, SLOPED)
    expected = -sum(pos[k] / 100 * s.loc[k, "pv01"] for k in pos)
    assert out["Rates only"]["pnl"][0] == pytest.approx(expected, rel=0.01)


TAPE = """<?xml version="1.0" encoding="UTF-8"?>
<assetData xmlns="http://www.sec.gov/edgar/document/absee/autoloan/assetdata">
{}
</assetData>"""
LOAN = """<assets><assetNumber>{id}</assetNumber><originationDate>03/2025</originationDate>
<reportingPeriodEndingDate>09-30-2026</reportingPeriodEndingDate>
<reportingPeriodBeginningLoanBalanceAmount>{beg}</reportingPeriodBeginningLoanBalanceAmount>
<reportingPeriodActualEndBalanceAmount>{end}</reportingPeriodActualEndBalanceAmount>
<reportingPeriodInterestRatePercentage>0.06</reportingPeriodInterestRatePercentage>
<remainingTermToMaturityNumber>40</remainingTermToMaturityNumber>
<scheduledPrincipalAmount>{sched}</scheduledPrincipalAmount>
<actualPrincipalCollectedAmount>{coll}</actualPrincipalCollectedAmount>
{extra}<currentDelinquencyStatus>0</currentDelinquencyStatus></assets>"""


def test_tape_stats_on_a_small_tape(tmp_path: Path):
    loans = [dict(id=1, beg=10000, end=9600, sched=400, coll=400, extra=""),          # on schedule
             dict(id=2, beg=10000, end=0, sched=400, coll=10000,                      # pays off
                  extra="<zeroBalanceCode>1</zeroBalanceCode>"),
             dict(id=3, beg=20000, end=0, sched=500, coll=12000,                      # charged off
                  extra="<chargedoffPrincipalAmount>8000</chargedoffPrincipalAmount>"
                        "<recoveredAmount>0</recoveredAmount><zeroBalanceCode>4</zeroBalanceCode>")]
    xml = "".join(LOAN.format(**x) for x in loans) + \
        "<assets><assetNumber>9</assetNumber><recoveredAmount>500</recoveredAmount></assets>"
    path = tmp_path / "tape.xml"
    path.write_text(TAPE.format(xml))
    s = tape_stats(parse_tape(path))
    assert s["loans"] == 3 and s["begin_balance"] == 40000 and s["end_balance"] == 9600
    assert s["prepay"] == pytest.approx(9600)                       # 10,000 paid less 400 due
    assert s["smm"] == pytest.approx(9600 / (40000 - 20000 - 800))
    assert s["mdr"] == pytest.approx(0.5)
    assert s["chargeoff"] - s["recoveries"] == pytest.approx(7500)
    assert s["wa_age"] == pytest.approx(19)                          # Mar 2025 -> Sep 2026, +1
    lines = rep_lines(parse_tape(path))
    assert lines["balance"].sum() == pytest.approx(9600) and lines["term"].iloc[0] == 40
