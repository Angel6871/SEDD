#!/usr/bin/env python3
"""
Compare equilibrium and frozen-throat CEA run CSV outputs.

This script is intentionally independent from config.yaml.
It reads two CSV files (defaults to the standard output folders), aligns rows
by operating point, and prints report-ready numeric deltas.

Usage:
    python3 compare_eq_frozen.py
    python3 compare_eq_frozen.py --eq outputs/equilibrium_run/results.csv --frozen outputs/frozen_throat_run/results.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_EQ = "outputs/equilibrium_run/results.csv"
#DEFAULT_FROZEN = "outputs/frozen_throat_run/results.csv"
DEFAULT_FROZEN = "outputs/frozen_throat/results.csv"

# Primary alignment keys (must identify a unique operating point).
KEY_FIELDS = ("Pc_bar", "OF")

# Columns worth reporting for chemistry/state comparison only.
REPORT_COLUMNS = (
    "Tc_K",
    "cstar_m_s",
    "mw_kg_kmol",
    "gamma",
    "gammas",
    "phi",
)

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare equilibrium vs frozen-throat results CSV files")
    p.add_argument("--eq", default=DEFAULT_EQ, help=f"equilibrium CSV path (default: {DEFAULT_EQ})")
    p.add_argument("--frozen", default=DEFAULT_FROZEN, help=f"frozen-throat CSV path (default: {DEFAULT_FROZEN})")
    return p.parse_args()


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def key_tuple(row: Dict[str, str], keys: Sequence[str]) -> Tuple[str, ...]:
    return tuple((row.get(k, "") or "").strip() for k in keys)


def build_index(rows: Iterable[Dict[str, str]], keys: Sequence[str]) -> Dict[Tuple[str, ...], Dict[str, str]]:
    idx: Dict[Tuple[str, ...], Dict[str, str]] = {}
    for row in rows:
        k = key_tuple(row, keys)
        if k in idx:
            raise ValueError(f"Duplicate key {k} in input CSV; expected unique operating points.")
        idx[k] = row
    return idx


def fmt_num(x: Optional[float], digits: int = 6) -> str:
    if x is None:
        return "n/a"
    if x == 0:
        return "0"
    return f"{x:.{digits}g}"


def fmt_pct(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "n/a"
    return f"{x:.{digits}f}%"


def signed(x: float, digits: int = 6) -> str:
    if x == 0:
        return "0"
    return f"{x:+.{digits}g}"


def summarize_column(
    col: str,
    eq_rows: Dict[Tuple[str, ...], Dict[str, str]],
    fr_rows: Dict[Tuple[str, ...], Dict[str, str]],
    matched: Sequence[Tuple[str, ...]],
) -> Optional[Dict[str, object]]:
    deltas: List[float] = []
    abs_deltas: List[float] = []
    rel_pcts: List[float] = []

    max_abs_delta: Optional[float] = None
    max_abs_key: Optional[Tuple[str, ...]] = None

    eq_vals: List[float] = []
    fr_vals: List[float] = []

    for k in matched:
        eq_v = to_float(eq_rows[k].get(col))
        fr_v = to_float(fr_rows[k].get(col))
        if eq_v is None or fr_v is None:
            continue

        d = fr_v - eq_v
        ad = abs(d)

        eq_vals.append(eq_v)
        fr_vals.append(fr_v)
        deltas.append(d)
        abs_deltas.append(ad)

        if abs(eq_v) > EPS:
            rel_pcts.append((d / eq_v) * 100.0)

        if max_abs_delta is None or ad > max_abs_delta:
            max_abs_delta = ad
            max_abs_key = k

    if not deltas:
        return None

    mean_delta = mean(deltas)
    mean_abs_delta = mean(abs_deltas)
    max_rel_pct = max((abs(v) for v in rel_pcts), default=None)

    return {
        "col": col,
        "n": len(deltas),
        "eq_mean": mean(eq_vals),
        "fr_mean": mean(fr_vals),
        "mean_delta": mean_delta,
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": max_abs_delta,
        "max_abs_key": max_abs_key,
        "max_abs_rel_pct": max_rel_pct,
    }


def point_str(k: Tuple[str, ...]) -> str:
    return ", ".join(f"{f}={v}" for f, v in zip(KEY_FIELDS, k))


def main() -> None:
    args = parse_args()

    eq = read_csv_rows(args.eq)
    fr = read_csv_rows(args.frozen)

    eq_idx = build_index(eq, KEY_FIELDS)
    fr_idx = build_index(fr, KEY_FIELDS)

    eq_keys = set(eq_idx)
    fr_keys = set(fr_idx)
    matched = sorted(eq_keys & fr_keys)

    if not matched:
        raise RuntimeError("No overlapping operating points between equilibrium and frozen files.")

    eq_only = sorted(eq_keys - fr_keys)
    fr_only = sorted(fr_keys - eq_keys)

    print("=== Equilibrium vs Frozen-Throat Numeric Comparison ===")
    print(f"EQ file     : {args.eq}")
    print(f"Frozen file : {args.frozen}")
    print(f"Matched operating points: {len(matched)}")
    if eq_only:
        print(f"Only in EQ ({len(eq_only)}): {point_str(eq_only[0])}")
    if fr_only:
        print(f"Only in Frozen ({len(fr_only)}): {point_str(fr_only[0])}")

    print("\nDelta definition: frozen_throat - fulleq")
    print(
        "Column".ljust(12),
        "MeanDelta".rjust(14),
        "MeanAbsDelta".rjust(14),
        "MaxAbsDelta".rjust(14),
        "Max|Rel|".rjust(10),
        sep=" | ",
    )
    print("-" * 74)

    summaries: List[Dict[str, object]] = []
    for c in REPORT_COLUMNS:
        s = summarize_column(c, eq_idx, fr_idx, matched)
        if s is None:
            continue
        summaries.append(s)
        print(
            str(s["col"]).ljust(12),
            signed(float(s["mean_delta"]), 6).rjust(14),
            fmt_num(float(s["mean_abs_delta"]), 6).rjust(14),
            fmt_num(float(s["max_abs_delta"]), 6).rjust(14),
            fmt_pct(s["max_abs_rel_pct"], 4).rjust(10),
            sep=" | ",
        )

    # Report-ready headline metrics for chemistry/state outputs.
    def get_summary(col: str) -> Optional[Dict[str, object]]:
        for s in summaries:
            if s["col"] == col:
                return s
        return None


if __name__ == "__main__":
    main()
