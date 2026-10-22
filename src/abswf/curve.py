"""Treasury par curve: interpolation, a bootstrapped zero curve, and data loading.

Par yields are Treasury constant-maturity yields on a bond-equivalent (semiannual)
basis. Bills (1 year and under) are treated as zero-coupon; longer tenors are
bootstrapped as semiannual par bonds on a half-year grid.
"""
from __future__ import annotations

import io
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TENORS = {"1 Mo": 1 / 12, "2 Mo": 2 / 12, "3 Mo": 0.25, "4 Mo": 1 / 3, "6 Mo": 0.5,
          "1 Yr": 1.0, "2 Yr": 2.0, "3 Yr": 3.0, "5 Yr": 5.0, "7 Yr": 7.0,
          "10 Yr": 10.0, "20 Yr": 20.0, "30 Yr": 30.0}
TREASURY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
                "&field_tdr_date_value={year}&page&_format=csv")


@dataclass(frozen=True)
class Curve:
    tenors: np.ndarray
    par: np.ndarray            # decimal par yields at `tenors` (years)
    asof: str = ""
    _grid: tuple = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        grid = np.arange(1, 61) * 0.5
        c = self.par_yield(grid)
        df = np.empty_like(grid)
        for i, (t, y) in enumerate(zip(grid, c)):
            df[i] = (1 + y / 2) ** (-2 * t) if t <= 1 else (1 - y / 2 * df[:i].sum()) / (1 + y / 2)
        object.__setattr__(self, "_grid", (grid, 2 * (df ** (-1 / (2 * grid)) - 1)))

    @classmethod
    def flat(cls, y: float) -> "Curve":
        t = np.array(list(TENORS.values()))
        return cls(t, np.full(len(t), y), "flat")

    @classmethod
    def from_row(cls, row: pd.Series) -> "Curve":
        row = row.dropna()
        cols = [c for c in TENORS if c in row.index]
        return cls(np.array([TENORS[c] for c in cols]), row[cols].to_numpy(float) / 100,
                   str(getattr(row, "name", "")))

    def par_yield(self, t):
        return np.interp(t, self.tenors, self.par)

    def shifted(self, bp: float = 0.0, by_tenor: np.ndarray | None = None) -> "Curve":
        par = self.par + bp / 1e4 + (0 if by_tenor is None else np.asarray(by_tenor))
        return Curve(self.tenors, par, self.asof)

    def zero(self, t):
        grid, z = self._grid
        t = np.asarray(t, float)
        return np.where(t < 0.5, self.par_yield(t), np.interp(t, grid, z))

    def discount(self, t, spread: float = 0.0):
        t = np.asarray(t, float)
        return (1 + (self.zero(t) + spread) / 2) ** (-2 * t)


def fetch_treasury(years, path: Path, user_agent: str, pause: float = 1.0) -> pd.DataFrame:
    """Download daily par yield curves from Treasury.gov for the given years."""
    frames = []
    for y in years:
        req = urllib.request.Request(TREASURY_URL.format(year=y), headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=60) as r:
            frames.append(pd.read_csv(io.BytesIO(r.read())))
        time.sleep(pause)
    df = pd.concat(frames)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()[[c for c in TENORS if c in df.columns]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
# ICE BofA US corporate OAS by rating, used as spread-change proxies for each class
SPREAD_PROXIES = {"A": "BAMLC0A1CAAA", "B": "BAMLC0A2CAA", "C": "BAMLC0A3CA", "D": "BAMLC0A4CBBB"}


def fetch_fred(series: dict, path: Path, user_agent: str, pause: float = 1.0) -> pd.DataFrame:
    """Daily FRED series (percent), one column per key of `series`."""
    cols = {}
    for key, sid in series.items():
        req = urllib.request.Request(FRED_URL.format(sid=sid), headers={"User-Agent": user_agent})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
        if not raw.lstrip().startswith(b"observation_date") and not raw.lstrip().startswith(b"DATE"):
            raise RuntimeError(f"FRED returned no CSV for {sid} (site down?)")
        s = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True, na_values=".").iloc[:, 0]
        cols[key] = s
        time.sleep(pause)
    df = pd.DataFrame(cols).dropna(how="all")
    df.index.name = "Date"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def load_treasury(path: Path) -> pd.DataFrame:
    """Cached curve history, with basic data-quality checks."""
    df = pd.read_csv(path, index_col="Date", parse_dates=True).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    bad = (df < -1) | (df > 25)
    if bad.any().any():
        raise ValueError(f"implausible yields on {df.index[bad.any(axis=1)].tolist()[:5]}")
    return df
