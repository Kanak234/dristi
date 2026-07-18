"""
DRISHTI — Natural Language Query Engine (fully offline)
=======================================================
A deterministic keyword/regex parser — no external API, no internet —
so it can run on air-gapped police infrastructure.

Input : "vehicle theft hotspots in koramangala last 3 months"
Output: {intent, crime_types, areas, start, end, interpretation}

Intents:
  map      -> where / hotspot / show on map
  trend    -> trend / pattern / over time / monthly
  compare  -> compare / vs / versus / rank / safest / worst / top
  count    -> how many / count / total / number of
"""

import re
from datetime import timedelta
import pandas as pd

CRIME_SYNONYMS = {
    "Vehicle Theft":    ["vehicle theft", "car theft", "bike theft",
                         "two wheeler", "vehicle stolen", "auto theft"],
    "Chain Snatching":  ["chain snatching", "chain snatch", "gold chain"],
    "Mobile Snatching": ["mobile snatching", "phone snatching",
                         "mobile theft", "phone theft"],
    "House Burglary":   ["burglar", "house break", "housebreak",
                         "break-in", "break in", "theft at home"],
    "Robbery":          ["robbery", "dacoity", "loot"],
    "Assault":          ["assault", "fight", "attack", "violence"],
    "Cybercrime Fraud": ["cyber", "online fraud", "upi fraud", "otp fraud",
                         "phishing", "internet fraud", "cyber crime",
                         "cybercrime"],
    "Cheating":         ["cheating", "fraud case", "scam"],
    "Drug Offense":     ["drug", "narcotic", "ganja", "peddling"],
    "Harassment":       ["harassment", "eve teasing", "eve-teasing",
                         "stalking"],
}

INTENT_PATTERNS = [
    ("compare", r"\b(compare|versus|\bvs\b|rank|ranking|safest|most "
                r"dangerous|worst|top \d+|top areas?|which areas?)\b"),
    ("trend",   r"\b(trend|pattern|over time|monthly|weekly|seasonal|"
                r"increase|decrease|rising|growth|by month|by hour|"
                r"time of day)\b"),
    ("count",   r"\b(how many|count|total|number of)\b"),
    ("map",     r"\b(hotspot|hot spot|map|where|location|show me|area[s]? "
                r"with|concentrat)\b"),
]

MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}


def parse_query(q: str, df: pd.DataFrame) -> dict:
    ql = " " + q.lower().strip() + " "
    data_end = pd.to_datetime(df["DateTime"]).max()
    data_start = pd.to_datetime(df["DateTime"]).min()

    # ---- crime types ----------------------------------------------------
    crimes = []
    for crime, syns in CRIME_SYNONYMS.items():
        if any(s in ql for s in syns):
            crimes.append(crime)

    # ---- areas -----------------------------------------------------------
    areas = [a for a in df["Area"].unique() if a.lower() in ql]

    # ---- time window ----------------------------------------------------
    start, end, twords = data_start, data_end, "full data range"
    m = re.search(r"last\s+(\d+)\s*(day|week|month|year)s?", ql)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = n * {"day": 1, "week": 7, "month": 30, "year": 365}[unit]
        start = data_end - timedelta(days=days)
        twords = f"last {n} {unit}{'s' if n > 1 else ''}"
    elif re.search(r"last\s+month\b", ql):
        start, twords = data_end - timedelta(days=30), "last month"
    elif re.search(r"last\s+week\b", ql):
        start, twords = data_end - timedelta(days=7), "last week"
    elif re.search(r"last\s+year\b", ql):
        start, twords = data_end - timedelta(days=365), "last year"
    elif re.search(r"this\s+year\b", ql):
        start = pd.Timestamp(data_end.year, 1, 1)
        twords = f"{data_end.year} so far"
    else:
        ym = re.search(r"\b(20\d{2})\b", ql)
        mo = next((v for k, v in MONTHS.items() if k in ql), None)
        if ym and mo:
            y = int(ym.group(1))
            start = pd.Timestamp(y, mo, 1)
            end = (start + pd.offsets.MonthEnd(1))
            twords = f"{start.strftime('%B %Y')}"
        elif ym:
            y = int(ym.group(1))
            start, end = pd.Timestamp(y, 1, 1), pd.Timestamp(y, 12, 31)
            twords = str(y)
        elif mo:
            # most recent occurrence of that month in the data
            y = data_end.year if mo <= data_end.month else data_end.year - 1
            start = pd.Timestamp(y, mo, 1)
            end = start + pd.offsets.MonthEnd(1)
            twords = start.strftime("%B %Y")

    # ---- night/day qualifier --------------------------------------------
    hours = None
    if re.search(r"\b(night|midnight|late)\b", ql):
        hours, hword = [22, 23, 0, 1, 2, 3, 4, 5], " at night"
    elif re.search(r"\bmorning\b", ql):
        hours, hword = [6, 7, 8, 9, 10, 11], " in the morning"
    elif re.search(r"\bevening\b", ql):
        hours, hword = [17, 18, 19, 20, 21], " in the evening"
    else:
        hword = ""

    # ---- intent ----------------------------------------------------------
    intent = "map"
    for name, pat in INTENT_PATTERNS:
        if re.search(pat, ql):
            intent = name
            break

    interp = (f"Showing **{intent.upper()}** of "
              f"**{', '.join(crimes) if crimes else 'all crime types'}** in "
              f"**{', '.join(areas) if areas else 'all areas'}** — "
              f"{twords}{hword}.")

    return {"intent": intent, "crime_types": crimes, "areas": areas,
            "start": start, "end": end, "hours": hours,
            "interpretation": interp}


def apply_filters(df: pd.DataFrame, p: dict) -> pd.DataFrame:
    dt = pd.to_datetime(df["DateTime"])
    out = df[(dt >= p["start"]) & (dt <= p["end"])]
    if p["crime_types"]:
        out = out[out["Crime_Type"].isin(p["crime_types"])]
    if p["areas"]:
        out = out[out["Area"].isin(p["areas"])]
    if p["hours"]:
        out = out[out["Hour"].isin(p["hours"])]
    return out


EXAMPLE_QUERIES = [
    "Vehicle theft hotspots in last 3 months",
    "Compare cybercrime across areas this year",
    "Chain snatching trend in KR Market",
    "How many burglaries in Whitefield last 6 months",
    "Assault at night in Koramangala",
    "Top areas for robbery in 2025",
]
