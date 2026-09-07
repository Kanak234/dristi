"""
DRISHTI — Synthetic Crime Data Generator
=========================================
Generates a realistic (but fully SYNTHETIC) FIR-style crime dataset for
Bengaluru city, modelled on publicly known crime patterns:
  - spatial variation across 20 real police-station areas
  - crime-type specific hour-of-day profiles
  - weekday/weekend effects, festival-season spikes, monsoon dips
  - year-over-year growth in cybercrime

NO real FIR data is used anywhere. The pipeline in app.py works identically
on a real FIR export with the same columns.

Run:  python generate_data.py   ->  data/crime_records.csv
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------- areas ----
# (name, lat, lon, base_intensity 0-1, spatial spread in degrees)
AREAS = [
    ("Majestic",        12.9767, 77.5713, 1.00, 0.006),
    ("KR Market",       12.9634, 77.5855, 0.95, 0.005),
    ("Shivajinagar",    12.9857, 77.6057, 0.85, 0.006),
    ("Koramangala",     12.9352, 77.6245, 0.80, 0.009),
    ("Indiranagar",     12.9719, 77.6412, 0.72, 0.008),
    ("Marathahalli",    12.9569, 77.7011, 0.70, 0.009),
    ("Whitefield",      12.9698, 77.7500, 0.68, 0.011),
    ("Electronic City", 12.8452, 77.6602, 0.62, 0.011),
    ("BTM Layout",      12.9166, 77.6101, 0.60, 0.008),
    ("HSR Layout",      12.9116, 77.6474, 0.58, 0.008),
    ("Peenya",          13.0290, 77.5190, 0.58, 0.010),
    ("Yeshwanthpur",    13.0280, 77.5400, 0.55, 0.008),
    ("Hebbal",          13.0358, 77.5970, 0.52, 0.009),
    ("Rajajinagar",     12.9866, 77.5517, 0.50, 0.007),
    ("Frazer Town",     13.0007, 77.6081, 0.48, 0.006),
    ("MG Road",         12.9757, 77.6050, 0.55, 0.005),
    ("Jayanagar",       12.9308, 77.5838, 0.42, 0.008),
    ("Banashankari",    12.9255, 77.5468, 0.40, 0.008),
    ("Malleshwaram",    13.0033, 77.5703, 0.38, 0.007),
    ("Bommanahalli",    12.8994, 77.6182, 0.52, 0.008),
]

# ------------------------------------------------------------ crime types --
# (name, overall share, severity 1-5, 24h hourly weight profile)
def _profile(peaks):
    """Build a smooth 24h weight profile from (hour, weight) peak points."""
    w = np.full(24, 0.15)
    for h, amp, spread in peaks:
        for i in range(24):
            d = min(abs(i - h), 24 - abs(i - h))
            w[i] += amp * np.exp(-(d ** 2) / (2 * spread ** 2))
    return w / w.sum()

CRIME_TYPES = [
    ("Vehicle Theft",     0.20, 3, _profile([(23, 1.0, 2.5), (2, 0.9, 2.0)])),
    ("Chain Snatching",   0.11, 3, _profile([(7, 0.9, 1.5), (18, 1.0, 1.8)])),
    ("Mobile Snatching",  0.10, 2, _profile([(9, 0.7, 2.0), (19, 1.0, 2.0)])),
    ("House Burglary",    0.12, 4, _profile([(13, 0.9, 2.5), (2, 0.7, 1.8)])),
    ("Robbery",           0.07, 4, _profile([(22, 1.0, 2.0)])),
    ("Assault",           0.09, 4, _profile([(22, 1.0, 2.5), (0, 0.6, 1.5)])),
    ("Cybercrime Fraud",  0.14, 3, _profile([(12, 0.8, 3.0), (16, 0.9, 3.0)])),
    ("Cheating",          0.07, 2, _profile([(12, 0.8, 3.5)])),
    ("Drug Offense",      0.05, 3, _profile([(20, 1.0, 2.0)])),
    ("Harassment",        0.05, 2, _profile([(18, 1.0, 2.0)])),
]

# area-level multipliers for specific crime types (spatial character)
AREA_CRIME_BOOST = {
    ("Majestic", "Mobile Snatching"): 1.8,
    ("Majestic", "Chain Snatching"): 1.5,
    ("KR Market", "Chain Snatching"): 1.9,
    ("KR Market", "Mobile Snatching"): 1.6,
    ("Shivajinagar", "Robbery"): 1.5,
    ("Koramangala", "Vehicle Theft"): 1.5,
    ("Koramangala", "Cybercrime Fraud"): 1.4,
    ("Indiranagar", "House Burglary"): 1.4,
    ("Indiranagar", "Assault"): 1.3,
    ("Whitefield", "Cybercrime Fraud"): 1.9,
    ("Whitefield", "House Burglary"): 1.4,
    ("Electronic City", "Cybercrime Fraud"): 1.8,
    ("Electronic City", "Vehicle Theft"): 1.4,
    ("Marathahalli", "Vehicle Theft"): 1.6,
    ("HSR Layout", "House Burglary"): 1.5,
    ("HSR Layout", "Cybercrime Fraud"): 1.4,
    ("Peenya", "Assault"): 1.5,
    ("Peenya", "Vehicle Theft"): 1.3,
    ("Hebbal", "Vehicle Theft"): 1.3,
    ("MG Road", "Mobile Snatching"): 1.4,
    ("Bommanahalli", "Robbery"): 1.3,
}

STATUS = ["Under Investigation", "Chargesheeted", "Closed - Convicted",
          "Closed - Untraced"]
STATUS_P = [0.38, 0.27, 0.13, 0.22]

START = datetime(2024, 1, 1)
END = datetime(2026, 7, 15)
TARGET_RECORDS = 26000


def month_factor(month, crime):
    """Seasonality: festival-season property crime spike, monsoon street dip."""
    f = 1.0
    if month in (10, 11, 12):                       # Dasara/Deepavali season
        if crime in ("House Burglary", "Vehicle Theft", "Chain Snatching",
                     "Robbery", "Mobile Snatching"):
            f *= 1.30
    if month in (6, 7, 8, 9):                        # monsoon
        if crime in ("Chain Snatching", "Mobile Snatching", "Harassment"):
            f *= 0.82
    if month in (3, 4, 5):                           # summer evenings
        if crime in ("Assault", "Drug Offense"):
            f *= 1.12
    return f


def year_factor(year, crime):
    """Cybercrime grows ~22%/yr; street snatching slowly declines."""
    k = year - 2024
    if crime == "Cybercrime Fraud":
        return 1.22 ** k
    if crime in ("Chain Snatching", "Mobile Snatching"):
        return 0.93 ** k
    return 1.0 + 0.03 * k


def dow_factor(dow, crime):
    """Weekend effect (dow: 0=Mon .. 6=Sun)."""
    if dow >= 5 and crime in ("Assault", "Robbery", "Drug Offense",
                              "Harassment"):
        return 1.35
    if dow >= 5 and crime == "Cybercrime Fraud":
        return 0.85
    return 1.0


def main():
    n_days = (END - START).days + 1
    days = [START + timedelta(days=i) for i in range(n_days)]

    # ---- expected count per (day, area, crime) --------------------------
    base_daily = TARGET_RECORDS / n_days           # ~28 incidents/day citywide
    area_w = np.array([a[3] for a in AREAS]); area_w = area_w / area_w.sum()
    crime_w = np.array([c[1] for c in CRIME_TYPES])

    records = []
    fir_no = 1
    for day in days:
        dow, month, year = day.weekday(), day.month, day.year
        for ai, (aname, alat, alon, _, aspread) in enumerate(AREAS):
            for ci, (cname, _, sev, hprof) in enumerate(CRIME_TYPES):
                lam = (base_daily * area_w[ai] * crime_w[ci]
                       * AREA_CRIME_BOOST.get((aname, cname), 1.0)
                       * month_factor(month, cname)
                       * year_factor(year, cname)
                       * dow_factor(dow, cname))
                k = rng.poisson(lam)
                for _ in range(k):
                    hour = int(rng.choice(24, p=hprof))
                    minute = int(rng.integers(0, 60))
                    ts = day.replace(hour=hour, minute=minute)
                    lat = alat + rng.normal(0, aspread)
                    lon = alon + rng.normal(0, aspread)
                    records.append((
                        f"FIR-{ts.year}-{fir_no:06d}", ts, cname, aname,
                        round(lat, 6), round(lon, 6), sev,
                        rng.choice(STATUS, p=STATUS_P),
                    ))
                    fir_no += 1

    df = pd.DataFrame(records, columns=[
        "FIR_ID", "DateTime", "Crime_Type", "Area",
        "Latitude", "Longitude", "Severity", "Status"])
    df = df.sort_values("DateTime").reset_index(drop=True)
    df["Date"] = df["DateTime"].dt.date
    df["Hour"] = df["DateTime"].dt.hour
    df["Day_of_Week"] = df["DateTime"].dt.day_name()
    df["Month"] = df["DateTime"].dt.month
    df["Year"] = df["DateTime"].dt.year

    df.to_csv("data/crime_records.csv", index=False)
    print(f"Generated {len(df):,} synthetic records "
          f"({df['DateTime'].min()} -> {df['DateTime'].max()})")
    print(df["Crime_Type"].value_counts().to_string())


if __name__ == "__main__":
    main()
