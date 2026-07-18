"""Quick smoke tests: python test_core.py"""
import time
import pandas as pd

t0 = time.time()
df = pd.read_csv("data/crime_records.csv", parse_dates=["DateTime"])
print(f"[data] {len(df):,} rows, {df['Area'].nunique()} areas, "
      f"{df['Crime_Type'].nunique()} crime types  ({time.time()-t0:.1f}s)")

# ---- ML engine ----------------------------------------------------------
from ml_engine import HotspotPredictor, DemandForecaster, ClusterDetector

t = time.time()
hp = HotspotPredictor(df)
print(f"[hotspot] trained on {hp.n_train:,} rows, val MAE="
      f"{hp.val_mae:.4f} incidents/day-slot  ({time.time()-t:.1f}s)")
risk = hp.area_risk("Night (22:00-06:00)", dow=5, month=7)
print(risk.head(5).to_string(index=False))
assert risk["risk_score"].max() == 100.0 and len(risk) == 20

t = time.time()
fc = DemandForecaster(df)
f14 = fc.forecast(14)
print(f"[forecast] val MAE={fc.val_mae:.2f}/day, next-14d mean="
      f"{f14.mean():.1f}  ({time.time()-t:.1f}s)")
assert len(f14) == 14 and (f14 >= 0).all()

t = time.time()
cd = ClusterDetector(df)
print(f"[clusters] {len(cd.clusters)} clusters, "
      f"{int(cd.clusters['emerging'].sum()) if len(cd.clusters) else 0} "
      f"emerging  ({time.time()-t:.1f}s)")
print(cd.clusters.head(4).to_string(index=False))

# ---- NLQ ----------------------------------------------------------------
from nlq_engine import parse_query, apply_filters, EXAMPLE_QUERIES

for q in EXAMPLE_QUERIES + ["upi fraud in whitefield last 2 weeks at night"]:
    p = parse_query(q, df)
    sub = apply_filters(df, p)
    print(f"[nlq] '{q}' -> intent={p['intent']}, crimes={p['crime_types']}, "
          f"areas={p['areas']}, rows={len(sub)}")
    assert p["intent"] in ("map", "trend", "compare", "count")

# ---- patrol -------------------------------------------------------------
from patrol import allocate

alloc = allocate(risk, total_units=60)
print(alloc[["Area", "risk_score", "patrol_units", "priority"]]
      .head(6).to_string(index=False))
assert alloc["patrol_units"].sum() == 60
assert (alloc["patrol_units"] >= 1).all()

print(f"\nALL SMOKE TESTS PASSED  (total {time.time()-t0:.1f}s)")
