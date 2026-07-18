"""
DRISHTI — Patrol Allocation
===========================
Allocates N patrol units across areas in proportion to the ML-predicted
risk for a chosen shift/day, using the largest-remainder method with a
guaranteed minimum of 1 unit per area.

This is deliberately *zone-level* resource planning — no individual-level
prediction or profiling is performed anywhere in DRISHTI.
"""

import numpy as np
import pandas as pd


def allocate(risk_df: pd.DataFrame, total_units: int,
             min_per_area: int = 1) -> pd.DataFrame:
    r = risk_df.copy().reset_index(drop=True)
    n = len(r)
    total_units = max(total_units, n * min_per_area)

    base = np.full(n, min_per_area, dtype=int)
    remaining = total_units - base.sum()

    w = r["expected_incidents"].to_numpy(dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(n, 1 / n)

    exact = w * remaining
    floor = np.floor(exact).astype(int)
    leftover = remaining - floor.sum()
    order = np.argsort(-(exact - floor))          # largest remainders first
    extra = np.zeros(n, dtype=int)
    extra[order[:leftover]] = 1

    r["patrol_units"] = base + floor + extra
    r["share_pct"] = (100 * r["patrol_units"] / total_units).round(1)
    r["priority"] = pd.cut(r["risk_score"], [-1, 40, 70, 101],
                           labels=["ROUTINE", "ELEVATED", "CRITICAL"])
    return r.sort_values("patrol_units", ascending=False)
