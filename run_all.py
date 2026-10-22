"""Reproduce every table and figure in reports/.

Usage:
    python run_all.py              # use the cached data in data/
    python run_all.py --refresh    # re-download EDGAR, Treasury.gov and FRED data first
                                   # (set EDGAR_USER_AGENT="Your Name you@example.com")
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from abswf import report  # noqa: E402
from abswf.analytics import (breakeven_loss_mult, equity_yield, metrics,  # noqa: E402
                             price_at_spread, stress_grid, yt_table)
from abswf.collateral import Pool, project_pool  # noqa: E402
from abswf.curve import (SPREAD_PROXIES, TENORS, Curve, fetch_fred,  # noqa: E402
                         fetch_treasury, load_treasury)
from abswf.deals import (SYNTH_BASE, SYNTH_SPREADS, calibrated_assumptions,  # noqa: E402
                         fordo_deal, fordo_index, synthetic_deal)
from abswf.edgar import Edgar, build_dataset, download_deal, reconcile  # noqa: E402
from abswf.risk import Scenario, historical_var, run_stress, sensitivities  # noqa: E402
from abswf.waterfall import run_waterfall  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "reports"
RAW, FORDO = DATA / "raw", DATA / "fordo2026a"
CURVE_CSV, SPREAD_CSV = DATA / "treasury_par_curve.csv", DATA / "credit_spreads.csv"
FIRST_PAY = pd.Timestamp("2026-11-15")      # month 1 = next payment date after 10/15/2026

CPR_ROW = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30]
ABS_ROW = [0.005, 0.010, 0.015, 0.018, 0.020]
PREPAY_MULTS = [0.5, 0.75, 1.0, 1.5, 2.0]
LOSS_MULTS = [0, 1, 2, 4, 6, 8, 10]
POSITIONS = {"A": 50e6, "B": 10e6, "C": 10e6, "D": 5e6}    # face held, for stress and VaR
# Assumed spreads to the interpolated Treasury at WAL for FORDO's fixed-rate classes
# (illustrative, not market quotes); the floating A-2b is priced at par.
FORDO_SPREADS = {"A-2a": 0.0035, "A-3": 0.0045, "A-4": 0.0055, "B": 0.0075, "C": 0.0100}
# Historical windows: rate (and, if available, spread) moves are taken from the data;
# the collateral overlay is an assumption.
HISTORICAL = [("2008 Lehman", "2008-09-12", "2008-12-31", dict(loss_mult=3, severity=0.60, prepay_mult=0.8)),
              ("2020 COVID", "2020-02-21", "2020-03-23", dict(loss_mult=1.5, prepay_mult=0.8)),
              ("2022 Fed hikes", "2022-01-03", "2022-10-21", {})]
WIDEN = {"A": 50, "B": 100, "C": 150, "D": 250}


def refresh() -> None:
    ua = os.environ.get("EDGAR_USER_AGENT", "")
    print("Downloading EDGAR filings (about 1 GB of loan tapes)...")
    download_deal(Edgar(ua), RAW)
    build_dataset(RAW, FORDO)
    fetch_treasury([2008, 2020, 2022, 2024, 2025, 2026], CURVE_CSV, ua)
    try:
        fetch_fred(SPREAD_PROXIES, SPREAD_CSV, ua)
    except Exception as e:                       # FRED outages should not stop the build
        print(f"  FRED unavailable ({e}); keeping cached spreads if any")


def daily_changes(levels: pd.DataFrame) -> pd.DataFrame:
    """Day-over-day changes in percent-quoted levels, as decimals; drops the jumps
    across gaps in the history (e.g. between the separately downloaded years)."""
    lv = levels.dropna()
    gap = lv.index.to_series().diff().dt.days
    return (lv.diff() / 100)[gap <= 5]


def level_on(df: pd.DataFrame, date: str) -> pd.Series:
    return df.loc[:date].iloc[-1]


def scenarios(hist: pd.DataFrame, spreads: pd.DataFrame | None) -> list[Scenario]:
    out = []
    for name, start, end, overlay in HISTORICAL:
        move = (level_on(hist, end) - level_on(hist, start)).dropna() * 100     # bp
        move.index = [TENORS[c] for c in move.index]
        sp = {}
        if spreads is not None and spreads.index[0] <= pd.Timestamp(start):
            sp = ((level_on(spreads, end) - level_on(spreads, start)) * 100).to_dict()
        label = f"{name} ({start} to {end})" + ("" if sp else ", rates only")
        out.append(Scenario(label, "historical", curve_bp=move, spread_bp=sp, **overlay))
    out += [Scenario("Rates +100bp parallel", "hypothetical", rates_bp=100),
            Scenario("Rates -100bp parallel", "hypothetical", rates_bp=-100),
            Scenario("Spreads A/B/C/D +50/100/150/250bp", "hypothetical", spread_bp=WIDEN),
            Scenario("Prepay x0.5", "hypothetical", prepay_mult=0.5),
            Scenario("Prepay x2.0", "hypothetical", prepay_mult=2.0),
            Scenario("CDR x4, severity 60%", "hypothetical", loss_mult=4, severity=0.60),
            Scenario("Recession: rates -100, spreads widen x1.5, CDR x4, sev 60%, prepay x0.75",
                     "hypothetical", rates_bp=-100, spread_bp={k: 1.5 * v for k, v in WIDEN.items()},
                     loss_mult=4, severity=0.60, prepay_mult=0.75)]
    return out


def synthetic_section(curve, hist, spreads, md: list[str]) -> dict:
    deal, base = synthetic_deal(), SYNTH_BASE
    res = run_waterfall(deal, base)
    prices = {t.name: price_at_spread(res, t.name, t.balance, SYNTH_SPREADS[t.name], curve)
              for t in deal.tranches}
    report.paydown_chart(res, deal, OUT / "synth_paydown.png")
    cnl = res.deal["cum_net_loss"].iloc[-1]
    md += ["## 1. SYNTH 2026-1 (synthetic prime auto)", "",
           f"Pool ${deal.pool.total / 1e6:,.0f}mm, WAC {100 * deal.pool.wac:.2f}%, servicing 1.00%. "
           f"Base case: 1.40% ABS prepayment, CDR ramping 0.4% to 1.6% over 12 months, "
           f"severity 50%, 4-month recovery lag, 10% clean-up call. Base-case cumulative "
           f"net loss {100 * cnl:.2f}% of the original pool.", "",
           "![Paydown](synth_paydown.png)", ""]

    base_m = {}
    rows = []
    for t in deal.tranches:
        m = metrics(res.cash_flow(t.name), res.principal(t.name), prices[t.name], t.balance, curve)
        base_m[t.name] = m
        sub = 1 - sum(x.balance for x in deal.tranches[:deal.names.index(t.name) + 1]) / deal.pool.total
        rows.append({"class": t.name, "size ($mm)": t.balance / 1e6, "coupon (%)": 100 * t.coupon,
                     "subordination+OC (%)": 100 * sub, "price": m["price"],
                     "yield (%)": 100 * m["yield"], "WAL (yr)": m["wal"],
                     "mod dur": m["mod_dur"], "convexity": m["convexity"],
                     "dur-adj yield (%)": 100 * m["dur_adj_yield"],
                     "I-spread (bp)": 1e4 * m["i_spread"]})
    md += ["### Base-case tranche analytics", "",
           "Prices are set at assumed spreads to the interpolated Treasury at WAL "
           f"(A/B/C/D +{', +'.join(f'{1e4 * v:.0f}' for v in SYNTH_SPREADS.values())}bp); "
           f"Treasury curve as of {curve.asof[:10]}.", "",
           report.md_table(pd.DataFrame(rows).set_index("class")), ""]

    scen = {"1.40 ABS (base)": base} | {f"{100 * c:.0f} CPR": replace(base, prepay=c, prepay_type="CPR")
                                         for c in CPR_ROW}
    md += ["### YT-style tables (fixed price, prepay scenarios, base defaults)", ""]
    for t in deal.tranches:
        yt = yt_table(deal, scen, t.name, prices[t.name], curve, first_pay=FIRST_PAY)
        yt.index.name = f"Class {t.name}"
        yt.to_csv(OUT / f"yt_synth_{t.name}.csv")
        md += [report.md_table(yt), ""]

    grid = stress_grid(deal, base, prices, PREPAY_MULTS, LOSS_MULTS)
    grid.to_csv(OUT / "stress_grid_synth.csv", index=False)
    report.day_heatmap(grid, OUT / "stress_grid_day.png")
    at_base = grid[grid["prepay_mult"] == 1.0]
    tab = at_base.pivot(index="class", columns="loss_mult", values="yield") * 100
    tab.columns = [f"yield @ {c:g}x loss (%)" for c in tab.columns]
    tab["principal loss @ 10x (%)"] = 100 * at_base[at_base["loss_mult"] == 10].set_index("class")["principal_loss"]
    be = {t.name: breakeven_loss_mult(deal, base, t.name) for t in deal.tranches}
    tab["break-even loss mult"] = [be[n][0] for n in tab.index]
    tab["break-even CNL (%)"] = [100 * be[n][1] for n in tab.index]
    cnl_row = at_base.groupby("loss_mult")["cnl"].first() * 100
    eq = {lm: 100 * equity_yield(run_waterfall(deal, base.stressed(1, lm)), deal) for lm in LOSS_MULTS}
    md += ["### Nominal vs default-adjusted yields", "",
           "Yield at the base-case price as the loss stress rises (base prepay). "
           "The 0x column is the nominal (no-default) yield. Pool cumulative net loss by "
           "column: " + ", ".join(f"{k:g}x = {v:.2f}%" for k, v in cnl_row.items()) +
           ". Equity IRR (funding the initial OC and reserve at par): " +
           ", ".join(f"{k:g}x = {v:.1f}%" for k, v in eq.items()) + ".", "",
           report.md_table(tab), "",
           "### Duration-adjusted yield, prepay x loss stress", "",
           "![Stress grid](stress_grid_day.png)", ""]

    sens = sensitivities(deal, base, prices, curve)
    sens["PV01 / (D x P)"] = sens["pv01"] / sens["dur_x_price"]
    sens["PV01 per $1mm"] = sens["pv01"] * 1e4
    sens["CS01 per $1mm"] = sens["cs01"] * 1e4
    md += ["### PV01 and CS01 (bump-and-reprice, per 100 face)", "",
           report.md_table(sens[["price", "mod_dur", "pv01", "cs01", "dur_x_price",
                                 "PV01 / (D x P)", "PV01 per $1mm", "CS01 per $1mm"]], "{:,.4f}"), ""]

    stress = run_stress(deal, base, prices, curve, POSITIONS, scenarios(hist, spreads))
    stress["P&L"] = stress["P&L"].round().astype(int)
    md += ["### Stress scenarios (full revaluation; position: "
           + ", ".join(f"{k} ${v / 1e6:.0f}mm" for k, v in POSITIONS.items()) + " face)", "",
           report.md_table(stress.drop(columns="kind")), ""]

    rates = daily_changes(hist).iloc[-500:]
    rates.columns = [TENORS[c] for c in rates.columns]
    sp = None if spreads is None else daily_changes(spreads).loc[rates.index[0]:]
    var = historical_var(deal, base, prices, curve, POSITIONS, rates, sp)
    legs = [k for k in var if isinstance(var[k], dict)]
    vt = pd.DataFrame({k: {"1-day 99% VaR ($)": var[k]["VaR"], "1-day 99% ES ($)": var[k]["ES"]}
                       for k in legs}).T
    md += [f"### Historical-simulation VaR ({var['days']} days, {var['start']:%Y-%m-%d} to "
           f"{var['end']:%Y-%m-%d})", "",
           "Daily changes in every Treasury par tenor" +
           (", plus daily changes in ICE BofA corporate OAS by rating (AAA/AA/A/BBB as "
            "proxies for A/B/C/D)" if sp is not None else
            "; credit-spread history was unavailable when this report was built, so this "
            "is rates-only") + ", applied to today's curve with full revaluation.", "",
           report.md_table(vt, "{:,.0f}"), ""]
    return {"prices": prices, "metrics": base_m, "breakeven": be, "grid": grid, "sens": sens,
            "stress": stress, "var": var, "cnl": cnl, "equity": eq}


def fordo_section(curve, md: list[str]) -> dict:
    stats = pd.read_csv(FORDO / "tape_stats.csv")
    reports = pd.read_csv(FORDO / "investor_reports.csv")
    rec = reconcile(stats, reports).set_index("check")
    report.speeds_chart(stats, OUT / "fordo_speeds.png")
    a = calibrated_assumptions(stats)
    a_all = calibrated_assumptions(stats, months=None)

    cut = Pool.lines(pd.read_csv(FORDO / "replines_cutoff.csv").to_dict("records"))
    proj = project_pool(cut, a_all, cleanup_call=None)["end_balance"].iloc[:len(reports)]
    fit = pd.DataFrame({"model EOP pool ($mm)": proj.to_numpy() / 1e6,
                        "10-D EOP pool ($mm)": reports["pool_end"].to_numpy() / 1e6},
                       index=pd.Index(reports["period"], name="period"))
    fit["diff (%)"] = 100 * (fit.iloc[:, 0] / fit.iloc[:, 1] - 1)

    last = reports.iloc[-1]
    deal = fordo_deal(pd.read_csv(FORDO / "replines_latest.csv"), last)
    idx = fordo_index(last)
    res = run_waterfall(deal, a, idx)
    prices = {t.name: 100.0 if t.floating else
              price_at_spread(res, t.name, t.balance, FORDO_SPREADS[t.name], curve)
              for t in deal.tranches}

    sp = stats.set_index("period")
    speeds = pd.DataFrame({"loans": sp["loans"], "begin pool ($mm)": sp["begin_balance"] / 1e6,
                           "ABS (%)": 100 * sp["abs"], "CPR (%)": 100 * sp["cpr"],
                           "CDR (%)": 100 * sp["cdr"],
                           "net loss ($k)": (sp["chargeoff"] - sp["recoveries"]) / 1e3,
                           "60+ dq (%)": 100 * sp["dq60_pct"]})
    md += ["## 2. FORDO 2026-A, calibrated to public SEC filings", "",
           f"Seven monthly ABS-EE loan tapes (Mar-Sep 2026 collection periods, "
           f"{int(stats['loans'].iloc[0]):,} receivables at cutoff) and the matching 10-D "
           "investor reports.", "", "### Data checks: loan tape vs. 10-D investor report", "",
           report.md_table(rec, "{:,.4g}"), "",
           "Liquidation proceeds differ by definition: the tape measure counts principal "
           "collected on receivables charged off that month, while the 10-D also counts "
           "repossession proceeds received before charge-off.", "",
           "### Realized monthly performance (from the loan tapes)", "",
           report.md_table(speeds), "", "![Speeds](fordo_speeds.png)", "",
           f"Calibrated assumptions (trailing 4 months, balance-weighted): "
           f"{100 * a.prepay:.2f}% ABS, {100 * a.cdr:.2f}% CDR; severity "
           f"{100 * a.severity:.1f}% (cumulative net loss / gross defaults since cutoff); "
           f"recovery lag {a.recovery_lag} months (assumed).", "",
           "### In-sample check: cutoff pool projected with all-period average speeds", "",
           report.md_table(fit), ""]

    scen = {f"{100 * x:.2f} ABS": replace(a, prepay=x) for x in ABS_ROW[:2]}
    scen[f"{100 * a.prepay:.2f} ABS (calibrated)"] = a
    scen |= {f"{100 * x:.2f} ABS": replace(a, prepay=x) for x in ABS_ROW[2:]}
    yt = yt_table(deal, scen, "A-3", prices["A-3"], curve, idx, FIRST_PAY)
    yt.index.name = "Class A-3"
    yt.to_csv(OUT / "yt_fordo_A-3.csv")
    sens = sensitivities(deal, a, prices, curve, idx)
    be = {n: breakeven_loss_mult(deal, a, n, index=idx) for n in ("A-3", "A-4", "B", "C")}
    bet = pd.DataFrame({"break-even loss mult": {k: v[0] for k, v in be.items()},
                        "break-even CNL (%)": {k: 100 * v[1] for k, v in be.items()}})
    to_date = (stats["chargeoff"].sum() - stats["recoveries"].sum()) / reports["pool_begin"].iloc[0]
    future = res.deal["cum_net_loss"].iloc[-1]
    md += ["### Class A-3 YT-style table", "",
           f"Settlement on the 10/15/2026 payment date; prices at assumed spreads (A-2a/A-3/A-4/B/C "
           f"+{', +'.join(f'{1e4 * v:.0f}' for v in FORDO_SPREADS.values())}bp; A-2b at par); "
           f"SOFR held at {100 * idx:.3f}% (implied by the A-2b coupon).", "",
           report.md_table(yt), "",
           "### PV01 and CS01 (per 100 face); A-2b floats on SOFR", "",
           report.md_table(sens[["price", "mod_dur", "pv01", "cs01"]], "{:,.4f}"), "",
           "### Break-even loss multiples (calibrated base)", "",
           f"Net losses to date are {100 * to_date:.3f}% of the cutoff pool; the calibrated "
           f"base case projects a further {100 * future:.2f}%. Break-even CNL counts future "
           "losses only, as a share of the cutoff pool. Early-life default rates understate "
           "lifetime losses for auto pools, so the base case is not a loss forecast.", "",
           report.md_table(bet), ""]
    return {"assumptions": a, "fit": fit, "rec": rec, "sens": sens, "breakeven": be,
            "yt": yt, "stats": stats, "future_cnl": future, "to_date_cnl": to_date}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-download all data")
    args = ap.parse_args()
    if args.refresh:
        refresh()
    OUT.mkdir(exist_ok=True)
    hist = load_treasury(CURVE_CSV)
    spreads = pd.read_csv(SPREAD_CSV, index_col=0, parse_dates=True) if SPREAD_CSV.exists() else None
    curve = Curve.from_row(hist.iloc[-1])

    md = ["# ABS waterfall and tranche risk report", "",
          f"Generated by `python run_all.py`. Treasury par curve as of {curve.asof[:10]} "
          "(Treasury.gov).", ""]
    syn = synthetic_section(curve, hist, spreads, md)
    ford = fordo_section(curve, md)
    (OUT / "report.md").write_text("\n".join(md))

    print(f"Curve as of {curve.asof[:10]}")
    print(f"SYNTH base CNL {100 * syn['cnl']:.2f}%; break-even loss multiples: " +
          ", ".join(f"{k} {v[0]:.1f}x (CNL {100 * v[1]:.1f}%)" for k, v in syn["breakeven"].items()))
    for k, m in syn["metrics"].items():
        print(f"  {k}: price {m['price']:.3f} yield {100 * m['yield']:.2f}% WAL {m['wal']:.2f} "
              f"dur {m['mod_dur']:.2f} DAY {100 * m['dur_adj_yield']:.2f}%")
    print("  PV01/(D x P): " + ", ".join(f"{k} {v:.4f}" for k, v in syn["sens"]["PV01 / (D x P)"].items()))
    for k in [k for k in syn["var"] if isinstance(syn["var"][k], dict)]:
        print(f"  VaR {k}: {syn['var'][k]['VaR']:,.0f}  ES {syn['var'][k]['ES']:,.0f}")
    print(f"FORDO calibrated {ford['assumptions']}")
    print(f"  in-sample max |diff| {ford['fit']['diff (%)'].abs().max():.2f}%")
    print(f"Report written to {(OUT / 'report.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
