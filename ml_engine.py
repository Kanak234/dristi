"""
DRISHTI — ML Engine
===================
Three honest, fully-offline models (scikit-learn only):

1. HotspotPredictor  – HistGradientBoostingRegressor trained on historical
   incident *rates* per (area, crime type, shift, day-of-week, month).
   Outputs a 0-100 risk score per area for any chosen shift/day/month.

2. DemandForecaster  – gradient-boosted autoregressive model (lags 1/7/14 +
   rolling mean + calendar features) forecasting citywide daily incident
   volume 14 days ahead.

3. ClusterDetector   – DBSCAN over the most recent 90 days of incident
   coordinates; flags clusters whose density grew vs the previous 90 days
   ("emerging" hotspots).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_absolute_error

SHIFTS = {
    "Night (22:00-06:00)":     [22, 23, 0, 1, 2, 3, 4, 5],
    "Morning (06:00-12:00)":   [6, 7, 8, 9, 10, 11],
    "Afternoon (12:00-17:00)": [12, 13, 14, 15, 16],
    "Evening (17:00-22:00)":   [17, 18, 19, 20, 21],
}
SHIFT_NAMES = list(SHIFTS.keys())
DOW_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
             "Saturday", "Sunday"]


def hour_to_shift(hour: int) -> str:
    for name, hours in SHIFTS.items():
        if hour in hours:
            return name
    return SHIFT_NAMES[0]


# =========================================================================
class HotspotPredictor:
    def __init__(self, df: pd.DataFrame):
        self.areas = sorted(df["Area"].unique())
        self.crimes = sorted(df["Crime_Type"].unique())
        self.a2i = {a: i for i, a in enumerate(self.areas)}
        self.c2i = {c: i for i, c in enumerate(self.crimes)}
        self.model = None
        self.val_mae = None
        self._train(df)

    def _training_table(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["Shift"] = d["Hour"].map(hour_to_shift)
        d["DOW"] = pd.to_datetime(d["DateTime"]).dt.dayofweek

        # how many calendar days of each (dow, month) exist in the data
        cal = (pd.to_datetime(d["DateTime"]).dt.normalize()
               .drop_duplicates().to_frame("day"))
        cal["DOW"] = cal["day"].dt.dayofweek
        cal["Month"] = cal["day"].dt.month
        day_counts = (cal.groupby(["DOW", "Month"]).size()
                      .rename("n_days").reset_index())

        counts = (d.groupby(["Area", "Crime_Type", "Shift", "DOW", "Month"])
                  .size().rename("count").reset_index())

        # full grid so the model also learns true zeros
        grid = pd.MultiIndex.from_product(
            [self.areas, self.crimes, SHIFT_NAMES,
             range(7), range(1, 13)],
            names=["Area", "Crime_Type", "Shift", "DOW", "Month"]
        ).to_frame(index=False)
        grid = grid.merge(day_counts, on=["DOW", "Month"], how="inner")
        grid = grid.merge(counts,
                          on=["Area", "Crime_Type", "Shift", "DOW", "Month"],
                          how="left").fillna({"count": 0})
        grid["rate"] = grid["count"] / grid["n_days"]   # incidents per day
        return grid

    def _featurize(self, g: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            g["Area"].map(self.a2i).to_numpy(),
            g["Crime_Type"].map(self.c2i).to_numpy(),
            g["Shift"].map({s: i for i, s in enumerate(SHIFT_NAMES)})
             .to_numpy(),
            g["DOW"].to_numpy(),
            g["Month"].to_numpy(),
            (g["DOW"] >= 5).astype(int).to_numpy(),
        ])

    def _train(self, df: pd.DataFrame):
        grid = self._training_table(df)
        X = self._featurize(grid)
        y = grid["rate"].to_numpy()
        idx = np.random.default_rng(42).permutation(len(X))
        cut = int(0.85 * len(X))
        tr, va = idx[:cut], idx[cut:]
        self.model = HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.08, max_depth=6, random_state=42,
            categorical_features=[0, 1, 2])
        self.model.fit(X[tr], y[tr])
        self.val_mae = float(mean_absolute_error(y[va],
                                                 self.model.predict(X[va])))
        self.n_train = len(tr)

    def area_risk(self, shift: str, dow: int, month: int,
                  crime_types=None) -> pd.DataFrame:
        """Risk table for every area for the given context. 0-100 scaled."""
        crimes = crime_types or self.crimes
        rows = [(a, c, shift, dow, month) for a in self.areas for c in crimes]
        g = pd.DataFrame(rows, columns=["Area", "Crime_Type", "Shift",
                                        "DOW", "Month"])
        g["pred"] = np.clip(self.model.predict(self._featurize(g)), 0, None)
        per_area = g.groupby("Area")["pred"].sum().rename("expected_incidents")
        top_crime = (g.sort_values("pred", ascending=False)
                     .groupby("Area").first()["Crime_Type"]
                     .rename("dominant_crime"))
        out = pd.concat([per_area, top_crime], axis=1).reset_index()
        mx = out["expected_incidents"].max()
        out["risk_score"] = (100 * out["expected_incidents"] / mx if mx > 0
                             else 0).round(1)
        return out.sort_values("risk_score", ascending=False)


# =========================================================================
class DemandForecaster:
    LAGS = [1, 7, 14]

    def __init__(self, df: pd.DataFrame):
        daily = (df.groupby(pd.to_datetime(df["DateTime"]).dt.normalize())
                 .size().rename("y"))
        daily = daily.asfreq("D", fill_value=0)
        self.history = daily
        self.model, self.val_mae = self._train(daily)

    def _features(self, s: pd.Series) -> pd.DataFrame:
        f = pd.DataFrame({"y": s})
        for L in self.LAGS:
            f[f"lag{L}"] = s.shift(L)
        f["roll7"] = s.shift(1).rolling(7).mean()
        f["dow"] = s.index.dayofweek
        f["month"] = s.index.month
        f["weekend"] = (f["dow"] >= 5).astype(int)
        return f.dropna()

    def _train(self, daily: pd.Series):
        f = self._features(daily)
        X, y = f.drop(columns="y").to_numpy(), f["y"].to_numpy()
        cut = int(0.9 * len(X))
        m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                                          random_state=42)
        m.fit(X[:cut], y[:cut])
        mae = float(mean_absolute_error(y[cut:], m.predict(X[cut:])))
        m.fit(X, y)                      # refit on everything for deployment
        return m, mae

    def forecast(self, horizon: int = 14) -> pd.Series:
        s = self.history.copy()
        for _ in range(horizon):
            nxt = s.index[-1] + pd.Timedelta(days=1)
            row = np.array([[s.iloc[-L] for L in self.LAGS] +
                            [s.iloc[-7:].mean(), nxt.dayofweek, nxt.month,
                             int(nxt.dayofweek >= 5)]])
            s.loc[nxt] = max(0.0, float(self.model.predict(row)[0]))
        out = s.iloc[-horizon:]
        out.name = "forecast"
        return out


# =========================================================================
class ClusterDetector:
    """DBSCAN emerging-cluster detection over recent vs previous window."""

    def __init__(self, df: pd.DataFrame, window_days: int = 90,
                 eps_deg: float = 0.0075, min_samples: int = 20):
        dt = pd.to_datetime(df["DateTime"])
        end = dt.max()
        recent = df[dt > end - pd.Timedelta(days=window_days)]
        previous = df[(dt <= end - pd.Timedelta(days=window_days)) &
                      (dt > end - pd.Timedelta(days=2 * window_days))]
        self.clusters = self._detect(recent, previous, eps_deg, min_samples)

    @staticmethod
    def _detect(recent, previous, eps, ms) -> pd.DataFrame:
        if len(recent) < ms:
            return pd.DataFrame()
        X = recent[["Latitude", "Longitude"]].to_numpy()
        labels = DBSCAN(eps=eps, min_samples=ms).fit_predict(X)
        rows = []
        for lb in sorted(set(labels) - {-1}):
            mask = labels == lb
            sub = recent[mask]
            clat, clon = sub["Latitude"].mean(), sub["Longitude"].mean()
            # cluster's own radius = 90th pct member distance from centroid
            r2 = float(np.quantile((sub["Latitude"] - clat) ** 2 +
                                   (sub["Longitude"] - clon) ** 2, 0.9))
            # recent incidents inside that same radius (apples-to-apples)
            rec_d2 = ((recent["Latitude"] - clat) ** 2 +
                      (recent["Longitude"] - clon) ** 2)
            rec_n = int((rec_d2 <= r2).sum())
            if len(previous):
                d2 = ((previous["Latitude"] - clat) ** 2 +
                      (previous["Longitude"] - clon) ** 2)
                prev_n = int((d2 <= r2).sum())
            else:
                prev_n = 0
            growth = (rec_n - prev_n) / prev_n * 100 if prev_n else np.inf
            rows.append({
                "cluster_id": int(lb),
                "lat": round(float(clat), 5),
                "lon": round(float(clon), 5),
                "incidents_recent": int(len(sub)),
                "incidents_previous": prev_n,
                "growth_pct": round(float(growth), 1)
                if np.isfinite(growth) else None,
                "dominant_crime": sub["Crime_Type"].mode().iloc[0],
                "nearest_area": sub["Area"].mode().iloc[0],
                "emerging": bool(np.isinf(growth) or growth >= 15),
            })
        out = pd.DataFrame(rows)
        if len(out):
            out = out.sort_values("incidents_recent",
                                  ascending=False).reset_index(drop=True)
        return out
