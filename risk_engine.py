import pandas as pd
import numpy as np

# High-risk MCC codes (gambling, crypto, cash, forex)
HIGH_RISK_MCC = [
    '7995',  # gambling
    '6051',  # crypto / quasi-cash
    '6011',  # ATM cash
    '4829',  # wire transfer
    '6211',  # securities
    '5933',  # pawn shops
    '7801', '7802', '7994',  # online gambling variants
]

# High-risk countries (FATF grey/blacklist + high-risk jurisdictions)
HIGH_RISK_COUNTRIES = [
    'Iran', 'North Korea', 'Myanmar', 'Russia', 'Syria',
    'Yemen', 'Libya', 'Somalia', 'Sudan', 'Venezuela',
    'Cuba', 'Afghanistan', 'Haiti', 'Laos', 'Vanuatu'
]

HOME_COUNTRY = 'Singapore'
STRUCTURING_THRESHOLD = 10000   # USD equivalent
LATE_NIGHT_START = 22           # 10pm
LATE_NIGHT_END = 6              # 6am
VELOCITY_WINDOW_HOURS = 1       # transactions per hour
HIGH_VELOCITY_THRESHOLD = 5     # >5 tx/hour = flag

REQUIRED_COLUMNS = [
    'customer_id', 'transaction_datetime', 'merchant_country', 'MCC_code', 'amount'
]


def load_and_prepare(filepath) -> pd.DataFrame:
    """Load CSV (path or file-like) and standardise column types."""
    df = pd.read_csv(filepath)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")

    df['customer_id'] = df['customer_id'].astype(str).str.strip()
    df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'], errors='coerce')
    df = df.dropna(subset=['transaction_datetime'])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    df['hour'] = df['transaction_datetime'].dt.hour
    df['date'] = df['transaction_datetime'].dt.date
    df['MCC_code'] = df['MCC_code'].astype(str).str.strip()
    df['merchant_country'] = df['merchant_country'].astype(str).str.strip()
    return df


def compute_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each customer, compute behavioural risk features.
    Returns one row per customer with all signal columns.
    """
    features = []
    window_seconds = VELOCITY_WINDOW_HOURS * 3600

    for customer_id, group in df.groupby('customer_id'):
        group = group.sort_values('transaction_datetime')

        # --- VELOCITY: max transactions in any 1-hour window ---
        tx_times = group['transaction_datetime'].tolist()
        max_velocity = 0
        for t in tx_times:
            window = [x for x in tx_times if 0 <= (x - t).total_seconds() <= window_seconds]
            max_velocity = max(max_velocity, len(window))

        # --- STRUCTURING: transactions just below threshold ---
        structuring_count = len(group[
            (group['amount'] >= STRUCTURING_THRESHOLD * 0.85) &
            (group['amount'] < STRUCTURING_THRESHOLD)
        ])

        # --- LATE NIGHT ACTIVITY ---
        late_night = group[
            (group['hour'] >= LATE_NIGHT_START) |
            (group['hour'] < LATE_NIGHT_END)
        ]
        late_night_ratio = len(late_night) / len(group) if len(group) > 0 else 0

        # --- HIGH RISK MCC ---
        risky = group[group['MCC_code'].isin(HIGH_RISK_MCC)]
        risky_mcc_count = len(risky)
        risky_mcc_codes = risky['MCC_code'].unique().tolist()

        # --- CROSS BORDER ---
        if 'cross_border_flag' in group.columns:
            cross_border_count = int((pd.to_numeric(group['cross_border_flag'], errors='coerce') == 1).sum())
        else:
            cross_border_count = int((group['merchant_country'] != HOME_COUNTRY).sum())

        # --- HIGH RISK GEOGRAPHY ---
        high_risk_country_count = len(group[group['merchant_country'].isin(HIGH_RISK_COUNTRIES)])
        unique_countries = group['merchant_country'].nunique()

        # --- TRANSACTION SIZE ---
        avg_tx_size = group['amount'].mean()
        max_tx_size = group['amount'].max()
        large_tx_count = len(group[group['amount'] >= STRUCTURING_THRESHOLD])

        # --- DECLINED TRANSACTIONS ---
        if 'authorisation_outcome' in group.columns:
            declined_count = int((group['authorisation_outcome'].astype(str).str.lower() == 'declined').sum())
            declined_ratio = declined_count / len(group)
        else:
            declined_count = 0
            declined_ratio = 0.0

        features.append({
            'customer_id': customer_id,
            'total_transactions': len(group),
            'total_amount': round(group['amount'].sum(), 2),
            'avg_tx_size': round(avg_tx_size, 2),
            'max_tx_size': round(max_tx_size, 2),
            'large_tx_count': large_tx_count,
            'max_velocity_per_hour': max_velocity,
            'structuring_count': structuring_count,
            'late_night_ratio': round(late_night_ratio, 3),
            'risky_mcc_count': risky_mcc_count,
            'risky_mcc_codes': ', '.join(risky_mcc_codes) if risky_mcc_codes else 'None',
            'cross_border_count': cross_border_count,
            'high_risk_country_count': high_risk_country_count,
            'unique_countries': unique_countries,
            'declined_count': declined_count,
            'declined_ratio': round(declined_ratio, 3),
            'customer_tier': group['customer_tier'].iloc[0] if 'customer_tier' in group.columns else 'Unknown',
        })

    return pd.DataFrame(features)


def compute_risk_score(features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a composite risk score (0–100) from behavioural features.
    Each component is scored then weighted.
    """
    df = features.copy()

    # Velocity score (0–25)
    df['velocity_score'] = np.clip(df['max_velocity_per_hour'] / HIGH_VELOCITY_THRESHOLD * 25, 0, 25)

    # Geography score (0–20)
    df['geography_score'] = np.clip(
        (df['high_risk_country_count'] * 5) + (df['unique_countries'] * 0.5), 0, 20
    )

    # MCC score (0–20)
    df['mcc_score'] = np.clip(df['risky_mcc_count'] * 5, 0, 20)

    # Late night score (0–15)
    df['late_night_score'] = df['late_night_ratio'] * 15

    # Structuring score (0–20)
    df['structuring_score'] = np.clip(df['structuring_count'] * 10, 0, 20)

    # Composite score
    df['risk_score'] = (
        df['velocity_score'] +
        df['geography_score'] +
        df['mcc_score'] +
        df['late_night_score'] +
        df['structuring_score']
    ).round(1)

    # Segment
    df['risk_level'] = pd.cut(
        df['risk_score'],
        bins=[-1, 30, 60, 100],
        labels=['LOW', 'MEDIUM', 'HIGH']
    ).astype(str)

    return df.sort_values('risk_score', ascending=False)


def get_top_flagged(scored_df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return top N highest risk customers."""
    return scored_df[scored_df['risk_level'] == 'HIGH'].head(n)


def build_customer_signal_summary(row: pd.Series) -> str:
    """
    Build a plain-English signal summary for a single customer row.
    This gets passed to the LLM as context.
    """
    signals = []

    if row['max_velocity_per_hour'] >= HIGH_VELOCITY_THRESHOLD:
        signals.append(f"{row['max_velocity_per_hour']} transactions in a single hour (velocity spike)")

    if row['structuring_count'] > 0:
        signals.append(f"{row['structuring_count']} transactions just below the $10,000 reporting threshold (potential structuring)")

    if row['high_risk_country_count'] > 0:
        signals.append(f"{row['high_risk_country_count']} transactions to/from high-risk jurisdictions")

    if row['risky_mcc_count'] > 0:
        signals.append(f"{row['risky_mcc_count']} transactions with high-risk MCC codes ({row['risky_mcc_codes']})")

    if row['late_night_ratio'] > 0.3:
        signals.append(f"{round(row['late_night_ratio']*100)}% of activity during late-night hours (10pm–6am)")

    if row['cross_border_count'] > 0:
        signals.append(f"{row['cross_border_count']} cross-border transactions")

    if row['declined_ratio'] > 0.2:
        signals.append(f"High decline rate ({round(row['declined_ratio']*100)}%) suggesting probing behaviour")

    return '; '.join(signals) if signals else "No major signals detected."
