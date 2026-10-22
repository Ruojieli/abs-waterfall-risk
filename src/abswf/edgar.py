"""Ford Credit Auto Owner Trust 2026-A from SEC EDGAR: Reg AB II ABS-EE loan tapes
(Exhibit 102) and the monthly investor reports filed with Form 10-D.

EDGAR fair access: every request carries a User-Agent naming the requester and a
contact email (set EDGAR_USER_AGENT), and requests are spaced `pause` seconds apart,
well under the SEC's limit of 10 per second.
"""
from __future__ import annotations

import html
import json
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .collateral import mdr_to_cdr, smm_to_abs, smm_to_cpr

CIK = 2113024     # Ford Credit Auto Owner Trust 2026-A
NS = "{http://www.sec.gov/edgar/document/absee/autoloan/assetdata}"
FIELDS = {"assetNumber": "id", "originationDate": "orig_date",
          "reportingPeriodEndingDate": "period_end",
          "reportingPeriodBeginningLoanBalanceAmount": "begin",
          "reportingPeriodActualEndBalanceAmount": "end",
          "reportingPeriodInterestRatePercentage": "apr",
          "remainingTermToMaturityNumber": "rem_term",
          "scheduledPrincipalAmount": "sched", "actualPrincipalCollectedAmount": "collected",
          "chargedoffPrincipalAmount": "chargeoff", "recoveredAmount": "recovered",
          "zeroBalanceCode": "zb_code", "currentDelinquencyStatus": "dpd"}


class Edgar:
    def __init__(self, user_agent: str, pause: float = 0.25):
        if not user_agent or "@" not in user_agent:
            raise ValueError("EDGAR requires a User-Agent with a contact email, "
                             "e.g. EDGAR_USER_AGENT='Jane Doe jane@example.com'")
        self.ua, self.pause, self._last = user_agent, pause, 0.0

    def get(self, url: str) -> bytes:
        time.sleep(max(0.0, self._last + self.pause - time.time()))
        req = urllib.request.Request(url, headers={"User-Agent": self.ua,
                                                   "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        finally:
            self._last = time.time()

    def filings(self, cik: int, form: str) -> list[dict]:
        rec = json.loads(self.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
        rec = rec["filings"]["recent"]
        return [{"accession": a, "filed": f, "period": p, "doc": d}
                for a, f, p, d, fm in zip(rec["accessionNumber"], rec["filingDate"],
                                          rec["reportDate"], rec["primaryDocument"], rec["form"])
                if fm == form]

    def files(self, cik: int, accession: str) -> list[dict]:
        url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/"
        items = json.loads(self.get(url + "index.json"))["directory"]["item"]
        return [{"name": i["name"], "size": int(i["size"] or 0), "url": url + i["name"]}
                for i in items]


def download_deal(edgar: Edgar, raw_dir: Path, cik: int = CIK) -> None:
    """Monthly ABS-EE loan tapes and 10-D investor reports, saved by reporting period."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    for f in edgar.filings(cik, "ABS-EE"):
        if "initial" in f["doc"]:            # pre-closing tapes filed by the depositor
            continue
        out = raw_dir / f"abs-ee_{f['period']}.xml"
        if not out.exists():
            xml = max((x for x in edgar.files(cik, f["accession"]) if x["name"].endswith(".xml")),
                      key=lambda x: x["size"])     # Exhibit 102 is the large asset file
            out.write_bytes(edgar.get(xml["url"]))
    for f in edgar.filings(cik, "10-D"):
        out = raw_dir / f"10-d_{f['period']}.htm"
        if not out.exists():
            rep = next(x for x in edgar.files(cik, f["accession"]) if "investorrep" in x["name"])
            out.write_bytes(edgar.get(rep["url"]))


def parse_tape(path: Path) -> pd.DataFrame:
    """One row per receivable record in an ABS-EE asset data file (streamed)."""
    rows = []
    for _, el in ET.iterparse(path):
        if el.tag == NS + "assets":
            rows.append({v: el.findtext(NS + k) for k, v in FIELDS.items()})
            el.clear()
    df = pd.DataFrame(rows)
    num = ["begin", "end", "apr", "rem_term", "sched", "collected", "chargeoff",
           "recovered", "zb_code", "dpd"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    period = pd.to_datetime(df["period_end"].dropna().iloc[0], format="%m-%d-%Y")
    orig = pd.to_datetime(df["orig_date"], format="%m/%Y", errors="coerce")
    df["age"] = (period.year - orig.dt.year) * 12 + (period.month - orig.dt.month)
    df.attrs["period"] = period
    return df


def tape_stats(df: pd.DataFrame) -> dict:
    """Realized pool performance for one reporting period.

    A receivable charged off in the period (zero-balance code 4) is a default: its
    gross default amount is its beginning balance and the principal collected on it
    is liquidation proceeds. Small deficiencies written off on payoffs (code 1) count
    as losses but not defaults. For all other receivables, principal collected above
    scheduled principal, plus repurchases (code 3), is prepayment, and
    SMM = prepayments / (beginning balance - defaults - scheduled principal).
    """
    x = df.fillna({"begin": 0, "end": 0, "sched": 0, "collected": 0,
                   "chargeoff": 0, "recovered": 0, "dpd": 0})
    act = x[x["begin"] > 0]
    dflt = act["zb_code"] == 4
    live = act[~dflt]
    repurchased = np.where(live["zb_code"] == 3, live["end"], 0.0)
    sched = np.minimum(live["sched"], live["begin"])
    prepay = np.clip(live["collected"] + repurchased - sched, 0, None)
    beg, gross_default = act["begin"].sum(), act.loc[dflt, "begin"].sum()
    smm = prepay.sum() / (beg - gross_default - sched.sum())
    mdr = gross_default / beg
    w = act["begin"] / beg
    age = float((w * act["age"]).sum()) + 1       # age at this period's payment
    end = x.loc[(x["end"] > 0) & x["zb_code"].isna()]
    return {"period": df.attrs["period"].strftime("%Y-%m"), "loans": int(len(act)),
            "begin_balance": beg, "end_balance": end["end"].sum(),
            "sched_principal": sched.sum(), "prepay": prepay.sum(),
            "principal_collected": live.loc[live["zb_code"] != 3, "collected"].sum(),
            "gross_default": gross_default,
            "liquidation_proceeds": act.loc[dflt, "collected"].sum(),
            "chargeoff": x["chargeoff"].sum(), "recoveries": x["recovered"].sum(),
            "wa_apr": float((w * act["apr"]).sum()), "wa_rem_term": float((w * act["rem_term"]).sum()),
            "wa_age": age, "smm": smm, "cpr": float(smm_to_cpr(smm)),
            "abs": float(smm_to_abs(smm, age)), "mdr": mdr, "cdr": float(mdr_to_cdr(mdr)),
            "dq60_pct": end.loc[end["dpd"] > 60, "end"].sum() / end["end"].sum()}


def rep_lines(df: pd.DataFrame, at: str = "end",
              apr_edges=(0, 0.005, 0.03, 0.05, 0.07, 0.10, 1.0),
              term_edges=(0, 24, 36, 48, 60, 72, 120)) -> pd.DataFrame:
    """Stratify the tape into rep lines by APR and remaining term.

    at="end": loans outstanding after the period (projection starts next month).
    at="begin": the pool as it stood at the start of the period.
    """
    x = df.copy()
    if at == "end":
        x = x[(x["end"] > 0) & ~(x["chargeoff"].fillna(0) > 1) & (x["zb_code"].isna())]
        x["bal"], x["term"] = x["end"], x["rem_term"]
    else:
        x = x[x["begin"] > 0]
        x["bal"], x["term"], x["age"] = x["begin"], x["rem_term"] + 1, x["age"] - 1
    x["apr_b"] = pd.cut(x["apr"], apr_edges, right=False)
    x["term_b"] = pd.cut(x["term"].clip(lower=1), term_edges, right=True)
    g = x.groupby(["apr_b", "term_b"], observed=True)
    wavg = lambda c: g.apply(lambda d: np.average(d[c], weights=d["bal"]), include_groups=False)
    out = pd.DataFrame({"balance": g["bal"].sum(), "rate": wavg("apr"), "term": wavg("term"),
                        "age": wavg("age"), "loans": g.size()}).reset_index(drop=True)
    out["term"] = out["term"].round().astype(int).clip(lower=1)
    out["age"] = out["age"].round().astype(int).clip(lower=0)
    return out


# ---- 10-D monthly investor report ------------------------------------------------

def _report_lines(raw: str) -> list[tuple[str, str, list[float]]]:
    """(section, label, numbers) for each row of the investor report tables."""
    t = re.sub(r"<(br|/tr|/p|/div)[^>]*>", "\n", raw, flags=re.I)
    t = re.sub(r"</t[dh]>", " | ", t, flags=re.I)
    t = html.unescape(re.sub(r"<[^>]+>", "", t))
    out, section = [], ""
    for line in t.split("\n"):
        cells = [c.strip() for c in line.split("|") if c.strip() and c.strip() not in "$%*"]
        if not cells:
            continue
        m = re.match(r"^([IVX]+)\.\s", cells[0])
        if m:
            section = m.group(1)
        nums = [float(c.replace(",", "")) for c in cells[1:] if re.fullmatch(r"-?[\d,]+\.?\d*", c)]
        out.append((section, cells[0], nums))
    return out


def parse_investor_report(path: Path) -> dict:
    lines = _report_lines(path.read_text(errors="ignore"))
    get = lambda sec, label, i=0: next(n[i] for s, l, n in lines if s == sec and l == label and n)
    text = path.read_text(errors="ignore")
    period = re.search(r"Collection Period\s*</[^>]+>.*?([A-Z][a-z]+ \d{4})", text, re.S).group(1)
    notes = {l.replace(" Notes", ""): n[2] for s, l, n in lines
             if s == "V" and l.startswith("Class ") and len(n) >= 4}
    rates = {l.replace(" Notes", ""): n[1] / 100 for s, l, n in lines
             if s == "I" and l.startswith("Class ") and len(n) >= 2}
    return {"period": pd.Timestamp(period).strftime("%Y-%m"),
            "pool_begin": get("V", "Pool Balance", 0), "pool_end": get("V", "Pool Balance", 1),
            "loans_begin": get("V", "Number of Receivables Outstanding", 0),
            "loans_end": get("V", "Number of Receivables Outstanding", 1),
            "principal_collections": get("II", "Principal Collections"),
            "prepayments_in_full": get("II", "Prepayments in Full"),
            "liquidation_proceeds": get("II", "Liquidation Proceeds"),
            "recoveries": get("II", "Recoveries"),
            "interest_collections": get("II", "Interest Collections"),
            "chargeoffs": get("VIII", "Realized Loss (Charge-Offs)", 1),
            "reserve_end": get("VII", "Ending Reserve Account Balance"),
            "ysoc": get("VI", "Yield Supplement Overcollateralization Amount"),
            "target_oc": get("VI", "Targeted Overcollateralization Amount"),
            "wa_apr": get("V", "Weighted Average APR", 1) / 100,
            "notes_end": json.dumps(notes), "note_rates": json.dumps(rates)}


def build_dataset(raw_dir: Path, out_dir: Path) -> None:
    """Compact, committed CSVs derived from the raw filings (which are not committed)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    tapes = sorted(raw_dir.glob("abs-ee_*.xml"))
    stats = []
    for i, p in enumerate(tapes):
        df = parse_tape(p)
        stats.append(tape_stats(df))
        if i == 0:
            rep_lines(df, at="begin").to_csv(out_dir / "replines_cutoff.csv", index=False)
        if i == len(tapes) - 1:
            rep_lines(df, at="end").to_csv(out_dir / "replines_latest.csv", index=False)
    pd.DataFrame(stats).to_csv(out_dir / "tape_stats.csv", index=False)
    reports = [parse_investor_report(p) for p in sorted(raw_dir.glob("10-d_*.htm"))]
    pd.DataFrame(reports).to_csv(out_dir / "investor_reports.csv", index=False)


def reconcile(tape: pd.DataFrame, reports: pd.DataFrame) -> pd.DataFrame:
    """Loan-tape aggregates against the servicer's investor report, by period."""
    m = tape.merge(reports, on="period", suffixes=("_tape", "_10d"))
    checks = {"Pool balance, beginning": ("begin_balance", "pool_begin"),
              "Pool balance, ending": ("end_balance", "pool_end"),
              "Receivables, beginning": ("loans", "loans_begin"),
              "Charge-offs": ("chargeoff", "chargeoffs"),
              "Recoveries": ("recoveries_tape", "recoveries_10d"),
              "Liquidation proceeds": ("liquidation_proceeds_tape", "liquidation_proceeds_10d")}
    m["principal_10d"] = m["principal_collections"] + m["prepayments_in_full"]
    checks["Principal collected (ex-defaults)"] = ("principal_collected", "principal_10d")
    rows = []
    for name, (a, b) in checks.items():
        diff = (m[a] - m[b]).abs()
        rows.append({"check": name, "max_abs_diff": diff.max(),
                     "max_rel_diff": (diff / m[b].abs().replace(0, np.nan)).max()})
    return pd.DataFrame(rows)
