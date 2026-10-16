"""Generate a synthetic transactions dataset that exercises every risk signal.

Usage:  python generate_sample_data.py   (writes data/sample_transactions.csv)

50 customers: most behave normally; a handful each exhibit one signal
(velocity, structuring, high-risk geography, high-risk MCC, late-night,
card probing), and a few combine several signals so they score HIGH.
"""
import os
import random
from datetime import datetime, timedelta

import pandas as pd

SEED = 42
START = datetime(2026, 9, 1)
DAYS = 30
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_transactions.csv")

# (merchant_category, MCC, min_amount, max_amount)
EVERYDAY = [
    ("grocery", "5411", 15, 220), ("restaurant", "5812", 12, 160),
    ("fast_food", "5814", 5, 30), ("transport", "4111", 2, 25),
    ("department_store", "5311", 30, 600), ("pharmacy", "5912", 8, 90),
    ("electronics", "5732", 80, 2500), ("clothing", "5651", 25, 400),
    ("streaming", "4899", 10, 25), ("airline", "4511", 150, 1800),
    ("hotel", "7011", 120, 900), ("misc_retail", "5999", 10, 300),
]
HIGH_RISK = {
    "gambling": ("gambling", "7995"), "crypto": ("quasi_cash", "6051"),
    "atm": ("atm_cash", "6011"), "wire": ("wire_transfer", "4829"),
    "securities": ("securities", "6211"), "pawn": ("pawn_shop", "5933"),
    "online_gambling": ("online_gambling", "7801"),
}
TRAVEL_COUNTRIES = ["Malaysia", "Thailand", "Indonesia", "Japan", "Australia",
                    "United Kingdom", "United States", "Hong Kong", "India"]
HIGH_RISK_COUNTRIES = ["Iran", "Myanmar", "Russia", "North Korea", "Syria", "Venezuela"]

rng = random.Random(SEED)
rows = []


def ts(day=None, hour=None, minute=None):
    day = rng.randrange(DAYS) if day is None else day
    hour = rng.choice(range(8, 22)) if hour is None else hour
    minute = rng.randrange(60) if minute is None else minute
    return START + timedelta(days=day, hours=hour, minutes=minute, seconds=rng.randrange(60))


def add(cid, tier, when, country, channel, category, mcc, amount, outcome="approved"):
    cross_border = int(country != "Singapore")
    high_risk_merchant = int(mcc in {v[1] for v in HIGH_RISK.values()})
    if high_risk_merchant or country in HIGH_RISK_COUNTRIES:
        risk_flag = "HIGH"
    elif cross_border or amount >= 5000:
        risk_flag = "MEDIUM"
    else:
        risk_flag = "LOW"
    rows.append({
        "customer_id": cid, "transaction_datetime": when, "merchant_country": country,
        "channel": channel, "merchant_category": category, "MCC_code": mcc,
        "amount": round(amount, 2), "currency": "SGD" if country == "Singapore" else "USD",
        "authorisation_outcome": outcome, "customer_tier": tier,
        "cross_border_flag": cross_border, "high_risk_merchant_flag": high_risk_merchant,
        "risk_flag": risk_flag,
    })


def everyday(cid, tier, n, travel_prob=0.1):
    for _ in range(n):
        cat, mcc, lo, hi = rng.choice(EVERYDAY)
        country = rng.choice(TRAVEL_COUNTRIES) if rng.random() < travel_prob else "Singapore"
        channel = rng.choice(["pos", "pos", "online", "contactless"])
        outcome = "declined" if rng.random() < 0.03 else "approved"
        add(cid, tier, ts(), country, channel, cat, mcc, rng.uniform(lo, hi), outcome)


def risky(cid, tier, kind, when, country, amount, channel="online", outcome="approved"):
    cat, mcc = HIGH_RISK[kind]
    add(cid, tier, when, country, channel, cat, mcc, amount, outcome)


customers = [f"C{i:03d}" for i in range(1, 51)]
tiers = {c: rng.choice(["STANDARD", "STANDARD", "STANDARD", "PREMIUM"]) for c in customers}

# ── Multi-signal HIGH-risk customers ──────────────────────────────────────────
# C001: structuring + high-risk geography + gambling/wire, late at night, velocity burst
c = "C001"; tiers[c] = "STANDARD"
everyday(c, tiers[c], 4)
for i, (kind, country) in enumerate([("gambling", "Russia"), ("gambling", "Russia"),
                                     ("wire", "Myanmar"), ("wire", "Russia"),
                                     ("gambling", "Russia"), ("wire", "Myanmar")]):
    risky(c, tiers[c], kind, ts(day=10, hour=23, minute=i * 9), country, rng.uniform(8600, 9950))

# C002: crypto layering from Iran, late night, heavy declines (probing)
c = "C002"; tiers[c] = "STANDARD"
everyday(c, tiers[c], 3)
for i in range(7):
    risky(c, tiers[c], "crypto", ts(day=5, hour=2, minute=i * 7), rng.choice(["Iran", "Iran", "Syria"]),
          rng.uniform(2000, 6000), outcome="declined" if i % 2 == 0 else "approved")
for d in (12, 19):
    risky(c, tiers[c], "crypto", ts(day=d, hour=3), "Iran", rng.uniform(8700, 9900))

# C003: rapid-fire wires + structuring to multiple jurisdictions (PREMIUM tier)
c = "C003"; tiers[c] = "PREMIUM"
everyday(c, tiers[c], 6, travel_prob=0.3)
for i in range(6):
    risky(c, tiers[c], "wire", ts(day=15, hour=22, minute=i * 8),
          rng.choice(["Venezuela", "Russia", "Hong Kong"]), rng.uniform(8500, 9990))

# C004: ATM cash + pawn shop + securities across borders at night
c = "C004"; tiers[c] = "STANDARD"
everyday(c, tiers[c], 5)
for i, kind in enumerate(["atm", "atm", "atm", "pawn", "securities", "atm"]):
    risky(c, tiers[c], kind, ts(day=20, hour=1, minute=i * 9), rng.choice(["Myanmar", "Thailand"]),
          rng.uniform(800, 9900 if kind == "securities" else 3000), channel="atm" if kind == "atm" else "pos")
risky(c, tiers[c], "wire", ts(day=22, hour=23), "North Korea", 9400)

# C005: online gambling binge from high-risk country
c = "C005"; tiers[c] = "STANDARD"
everyday(c, tiers[c], 4)
for i in range(8):
    risky(c, tiers[c], rng.choice(["online_gambling", "gambling"]), ts(day=8, hour=0, minute=i * 6),
          "Russia", rng.uniform(300, 2500))
risky(c, tiers[c], "gambling", ts(day=9, hour=1), "Russia", 9200)

# ── Signal-pair customers ─────────────────────────────────────────────────────
# In each pair the first customer adds a second signal (scores MEDIUM); the second
# shows the signal alone (stays LOW overall but the signal is still detected).
LATE = [22, 23, 0, 1, 2, 3]

# Velocity: burst of small online purchases within one hour (C006 at night)
for c, hour in (("C006", 23), ("C007", 13)):
    everyday(c, tiers[c], 6)
    for i in range(7):
        cat, mcc, lo, hi = rng.choice(EVERYDAY[:4])
        add(c, tiers[c], ts(day=14, hour=hour, minute=i * 8), "Singapore", "online", cat, mcc, rng.uniform(lo, hi))

# Structuring: just-under-threshold payments (C008 via wire transfer MCC)
for c, use_wire in (("C008", True), ("C009", False)):
    everyday(c, tiers[c], 6)
    for d in (3, 9, 16):
        if use_wire:
            risky(c, tiers[c], "wire", ts(day=d), "Singapore", rng.uniform(8550, 9980))
        else:
            cat, mcc, _, _ = EVERYDAY[6]
            add(c, tiers[c], ts(day=d), "Singapore", "online", cat, mcc, rng.uniform(8550, 9980))

# High-risk geography (C010 also mostly late at night)
for c, night in (("C010", True), ("C011", False)):
    everyday(c, tiers[c], 4)
    for _ in range(4):
        cat, mcc, lo, hi = rng.choice(EVERYDAY)
        add(c, tiers[c], ts(hour=rng.choice(LATE) if night else None), rng.choice(HIGH_RISK_COUNTRIES),
            "pos", cat, mcc, rng.uniform(lo, hi))

# High-risk MCC (C012 gambling at night, C013 domestic crypto in daytime)
for c, kind, night in (("C012", "gambling", True), ("C013", "crypto", False)):
    everyday(c, tiers[c], 4)
    for _ in range(5):
        risky(c, tiers[c], kind, ts(hour=rng.choice(LATE) if night else None), "Singapore", rng.uniform(100, 1500))

# Late-night (C014 also withdraws cash from ATMs)
for c, atm in (("C014", True), ("C015", False)):
    for _ in range(8):
        cat, mcc, lo, hi = rng.choice(EVERYDAY)
        add(c, tiers[c], ts(hour=rng.choice(LATE)), "Singapore", "online", cat, mcc, rng.uniform(lo, hi))
    if atm:
        for _ in range(3):
            risky(c, tiers[c], "atm", ts(hour=rng.choice(LATE)), "Singapore", rng.uniform(200, 1000), channel="atm")
    everyday(c, tiers[c], 2)

# Card probing: many declines on small online purchases (C016 also in high-risk countries)
for c, hr_geo in (("C016", True), ("C017", False)):
    everyday(c, tiers[c], 4)
    for _ in range(6):
        cat, mcc, lo, hi = rng.choice(EVERYDAY)
        country = rng.choice(HIGH_RISK_COUNTRIES if hr_geo else TRAVEL_COUNTRIES)
        add(c, tiers[c], ts(), country, "online", cat, mcc, rng.uniform(1, 15), "declined")

# ── Normal customers ──────────────────────────────────────────────────────────
for c in customers[17:]:
    everyday(c, tiers[c], rng.randint(4, 10), travel_prob=rng.choice([0.0, 0.05, 0.15, 0.3]))

df = pd.DataFrame(rows).sort_values("transaction_datetime").reset_index(drop=True)
df.insert(0, "record_id", range(1, len(df) + 1))
df["transaction_datetime"] = df["transaction_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
df.to_csv(OUT_PATH, index=False)
print(f"Wrote {len(df)} transactions for {df['customer_id'].nunique()} customers to {OUT_PATH}")
