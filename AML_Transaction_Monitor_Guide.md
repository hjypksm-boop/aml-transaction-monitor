# AML Transaction Monitoring System — Complete Build Guide
### From VS Code to Working Submission

---

## OVERVIEW OF WHAT YOU'RE BUILDING

```
CSV Transactions
      ↓
Python Risk Engine     ← detects patterns deterministically
      ↓
RAG (FAISS)            ← grounds findings in AML policy
      ↓
Groq Llama 3.3         ← generates narratives & review questions
      ↓
Streamlit Dashboard    ← presents risk cards, tables, narratives
```

**Tech Stack:**
- Python 3.10+
- Streamlit (UI)
- Pandas (feature engineering)
- FAISS + LangChain (RAG)
- Groq API with Llama 3.3 (LLM)
- Sentence-Transformers (embeddings)

---

## PHASE 1 — PROJECT SETUP IN VS CODE

### Step 1: Create Your Project Folder

Open VS Code terminal (`Ctrl + `` ` ``) and run:

```bash
mkdir nium_aml_monitor
cd nium_aml_monitor
code .
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3: Install All Dependencies

Create a file called `requirements.txt`:

```txt
streamlit==1.35.0
pandas==2.2.2
numpy==1.26.4
groq==0.9.0
faiss-cpu==1.8.0
langchain==0.2.6
langchain-community==0.2.6
sentence-transformers==3.0.1
python-dotenv==1.0.1
plotly==5.22.0
openpyxl==3.1.4
```

Then install:

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Your Folder Structure

```
nium_aml_monitor/
├── app.py                  ← Streamlit entry point
├── risk_engine.py          ← Python pattern detection
├── rag_engine.py           ← FAISS + policy retrieval
├── llm_engine.py           ← Groq narrative generation
├── .env                    ← API keys (never commit this)
├── requirements.txt
├── data/
│   └── transactions.csv    ← your dataset goes here
└── policy_docs/
    └── aml_policy.txt      ← AML rules for RAG grounding
```

Create the folders:

```bash
mkdir data policy_docs
```

### Step 5: Set Up Your API Key

Create a `.env` file:

```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free Groq API key at: https://console.groq.com

Create a `.gitignore` file:

```
.env
venv/
__pycache__/
*.pyc
```

---

## PHASE 2 — THE RISK ENGINE (Core Logic)

This is the most important file. It computes real behavioural signals — no LLM involved here.

### Step 6: Build `risk_engine.py`

```python
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

STRUCTURING_THRESHOLD = 10000   # USD equivalent
LATE_NIGHT_START = 22           # 10pm
LATE_NIGHT_END = 6              # 6am
VELOCITY_WINDOW_HOURS = 1       # transactions per hour
HIGH_VELOCITY_THRESHOLD = 5     # >5 tx/hour = flag


def load_and_prepare(filepath: str) -> pd.DataFrame:
    """Load CSV and standardise column types."""
    df = pd.read_csv(filepath)
    df['transaction_datetime'] = pd.to_datetime(df['transaction_datetime'])
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    df['hour'] = df['transaction_datetime'].dt.hour
    df['date'] = df['transaction_datetime'].dt.date
    df['MCC_code'] = df['MCC_code'].astype(str).str.strip()
    df['merchant_country'] = df['merchant_country'].str.strip()
    return df


def compute_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each customer, compute behavioural risk features.
    Returns one row per customer with all signal columns.
    """
    features = []

    for customer_id, group in df.groupby('customer_id'):
        group = group.sort_values('transaction_datetime')

        # --- VELOCITY: max transactions in any 1-hour window ---
        tx_times = group['transaction_datetime'].tolist()
        max_velocity = 0
        for i, t in enumerate(tx_times):
            window = [x for x in tx_times if 0 <= (x - t).total_seconds() <= 3600]
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
        risky_mcc_count = len(group[group['MCC_code'].isin(HIGH_RISK_MCC)])
        risky_mcc_codes = group[group['MCC_code'].isin(HIGH_RISK_MCC)]['MCC_code'].unique().tolist()

        # --- CROSS BORDER ---
        cross_border_count = len(group[group.get('cross_border_flag', pd.Series([0]*len(group))) == 1]) \
            if 'cross_border_flag' in group.columns else \
            len(group[group['merchant_country'] != 'Singapore'])

        # --- HIGH RISK GEOGRAPHY ---
        high_risk_country_count = len(group[group['merchant_country'].isin(HIGH_RISK_COUNTRIES)])
        unique_countries = group['merchant_country'].nunique()

        # --- TRANSACTION SIZE ---
        avg_tx_size = group['amount'].mean()
        max_tx_size = group['amount'].max()
        large_tx_count = len(group[group['amount'] >= STRUCTURING_THRESHOLD])

        # --- DECLINED TRANSACTIONS ---
        if 'authorisation_outcome' in group.columns:
            declined_count = len(group[group['authorisation_outcome'].str.lower() == 'declined'])
            declined_ratio = declined_count / len(group)
        else:
            declined_count = 0
            declined_ratio = 0.0

        features.append({
            'customer_id': customer_id,
            'total_transactions': len(group),
            'total_amount': group['amount'].sum(),
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
    )

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
```

---

## PHASE 3 — THE RAG ENGINE (Policy Grounding)

### Step 7: Create Your AML Policy Document

Create `policy_docs/aml_policy.txt`:

```
AML MONITORING POLICY — NIUM PROGRAMME LEVEL

VELOCITY RULES:
- Flag any customer with more than 5 transactions in a 1-hour window.
- Flag customers with more than 20 transactions in a 24-hour period.
- Escalate accounts showing consistent velocity spikes across multiple days.

STRUCTURING:
- Flag transactions between $8,500 and $9,999 as potential structuring.
- Multiple structuring-pattern transactions by the same customer in 7 days require SAR consideration.
- Structuring is a federal offence under the Bank Secrecy Act and equivalent AML frameworks.

GEOGRAPHY:
- Flag any transaction to/from FATF blacklisted or grey-listed jurisdictions.
- Enhanced due diligence required for transactions involving Iran, North Korea, Myanmar, Russia, Syria.
- Cross-border transactions above $5,000 require additional scrutiny.

MCC CODES:
- MCC 7995 (gambling) requires enhanced monitoring.
- MCC 6051 (quasi-cash / crypto) flagged for all tiers below Premium.
- MCC 4829 (wire transfer) combined with cross-border flag = immediate escalation.
- Multiple high-risk MCC codes in same session = escalate.

LATE NIGHT ACTIVITY:
- Activity between 22:00–06:00 representing >30% of total transactions is anomalous.
- Combine with high-risk MCC or geography for escalation decision.

CUSTOMER TIERS:
- STANDARD tier customers with HIGH risk scores require immediate review.
- PREMIUM tier customers require enhanced ongoing monitoring even at MEDIUM risk.

PROGRAMME REVIEW — DELEGATED KYC:
- Where KYC is delegated to the client, Nium carries programme-level risk.
- Top-risk customers must be reviewed in programme calls.
- Client must confirm enhanced KYC was performed for all HIGH risk customers.
- Client must explain the business purpose of any flagged transaction clusters.
```

### Step 8: Build `rag_engine.py`

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os


def build_vector_store(policy_dir: str = "policy_docs") -> FAISS:
    """Load all policy docs and build FAISS index."""
    docs = []
    for filename in os.listdir(policy_dir):
        if filename.endswith('.txt'):
            loader = TextLoader(os.path.join(policy_dir, filename))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def retrieve_policy_context(vectorstore: FAISS, query: str, k: int = 3) -> str:
    """Retrieve the top-k relevant policy chunks for a given signal query."""
    results = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in results])
```

---

## PHASE 4 — THE LLM ENGINE (Narrative Generation)

### Step 9: Build `llm_engine.py`

```python
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_customer_narrative(
    customer_id: str,
    risk_score: float,
    risk_level: str,
    signal_summary: str,
    policy_context: str,
    customer_tier: str
) -> dict:
    """
    Generate a full compliance narrative for a flagged customer.
    Returns: narrative, rule_recommendations, review_questions
    """

    prompt = f"""
You are a senior AML compliance analyst at Nium, a fintech company.
You are reviewing a flagged customer and must produce a structured compliance briefing.

CUSTOMER DETAILS:
- Customer ID: {customer_id}
- Customer Tier: {customer_tier}
- Risk Score: {risk_score}/100
- Risk Level: {risk_level}

DETECTED BEHAVIOURAL SIGNALS:
{signal_summary}

RELEVANT AML POLICY:
{policy_context}

IMPORTANT CONTEXT:
Nium operates a "trust but verify" model — KYC is delegated to the card-issuing client.
Nium carries programme-level risk. The client manages their own customers.

Please produce the following, each clearly labelled:

1. RISK NARRATIVE (3–5 sentences): A professional compliance narrative explaining why this customer is flagged and what the risk indicators suggest about their behaviour.

2. MONITORING RULES (3–5 bullet points): Specific, operationalisable rules that should be applied to this customer or customer segment going forward. Express as plain English thresholds.

3. PROGRAMME REVIEW QUESTIONS (3–5 bullet points): Questions Nium should ask the client during the programme review call about this specific customer. Focus on the delegated KYC context and the flagged behaviours.

Be precise, professional, and concise. Do not use vague language.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )

    return {"raw": response.choices[0].message.content}


def generate_portfolio_rules(top_customers_signals: str) -> str:
    """
    Generate portfolio-level monitoring rules based on the top flagged customers.
    """
    prompt = f"""
You are a senior AML compliance analyst. Based on the following flagged customer signals
from a new Singapore card-issuing client portfolio, generate 5–7 portfolio-level
monitoring rules that Nium should implement immediately.

FLAGGED CUSTOMER SIGNALS:
{top_customers_signals}

Format each rule as:
RULE [N]: [Plain English description] — Threshold: [Specific measurable threshold]

Focus on rules that could be operationalised in a real compliance monitoring system.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=600
    )

    return response.choices[0].message.content
```

---

## PHASE 5 — THE STREAMLIT DASHBOARD

### Step 10: Build `app.py`

```python
import streamlit as st
import pandas as pd
import plotly.express as px
from risk_engine import (
    load_and_prepare, compute_customer_features,
    compute_risk_score, get_top_flagged, build_customer_signal_summary
)
from rag_engine import build_vector_store, retrieve_policy_context
from llm_engine import generate_customer_narrative, generate_portfolio_rules

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nium AML Monitor",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Nium Transaction Monitoring — Trust but Verify")
st.caption("UC1 | AML Risk Intelligence Dashboard")

# ─── Sidebar: Upload ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Input")
    uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])
    run_button = st.button("▶ Run Analysis", type="primary")
    st.divider()
    st.caption("Powered by Python Risk Engine + Groq Llama 3.3 + FAISS RAG")

# ─── Main Logic ────────────────────────────────────────────────────────────────
if uploaded_file and run_button:

    with st.spinner("Loading transactions..."):
        df = load_and_prepare(uploaded_file)

    with st.spinner("Computing behavioural risk features..."):
        features = compute_customer_features(df)
        scored = compute_risk_score(features)
        top_flagged = get_top_flagged(scored, n=5)

    with st.spinner("Building policy knowledge base..."):
        vectorstore = build_vector_store()

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    st.subheader("Portfolio Overview")
    col1, col2, col3, col4 = st.columns(4)

    high_count = len(scored[scored['risk_level'] == 'HIGH'])
    med_count  = len(scored[scored['risk_level'] == 'MEDIUM'])
    low_count  = len(scored[scored['risk_level'] == 'LOW'])
    cb_total   = scored['cross_border_count'].sum()

    col1.metric("🔴 High Risk Customers", high_count)
    col2.metric("🟡 Medium Risk Customers", med_count)
    col3.metric("🟢 Low Risk Customers", low_count)
    col4.metric("🌐 Total Cross-Border Tx", int(cb_total))

    st.divider()

    # ── Risk Distribution Chart ────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Risk Distribution")
        dist = scored['risk_level'].value_counts().reset_index()
        dist.columns = ['Risk Level', 'Count']
        fig = px.pie(dist, names='Risk Level', values='Count',
                     color='Risk Level',
                     color_discrete_map={'HIGH':'#e74c3c','MEDIUM':'#f39c12','LOW':'#2ecc71'})
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Top 10 Customers by Risk Score")
        top10 = scored.head(10)[['customer_id','risk_score','risk_level',
                                  'max_velocity_per_hour','risky_mcc_count',
                                  'cross_border_count','structuring_count']]
        st.dataframe(top10, use_container_width=True, hide_index=True)

    st.divider()

    # ── Full Customer Table ────────────────────────────────────────────────────
    with st.expander("📋 Full Customer Risk Table"):
        display_cols = ['customer_id','risk_score','risk_level','total_transactions',
                        'avg_tx_size','max_velocity_per_hour','structuring_count',
                        'risky_mcc_count','cross_border_count','late_night_ratio',
                        'high_risk_country_count','customer_tier']
        st.dataframe(scored[display_cols], use_container_width=True, hide_index=True)

    st.divider()

    # ── Top Flagged Customers — AI Narratives ──────────────────────────────────
    st.subheader("🚨 Top Flagged Customers — AI Compliance Narratives")

    all_signals = []

    for _, row in top_flagged.iterrows():
        customer_id = str(row['customer_id'])
        signal_summary = build_customer_signal_summary(row)
        all_signals.append(f"Customer {customer_id}: {signal_summary}")

        policy_context = retrieve_policy_context(
            vectorstore,
            query=signal_summary
        )

        with st.spinner(f"Generating narrative for Customer {customer_id}..."):
            result = generate_customer_narrative(
                customer_id=customer_id,
                risk_score=row['risk_score'],
                risk_level=str(row['risk_level']),
                signal_summary=signal_summary,
                policy_context=policy_context,
                customer_tier=str(row['customer_tier'])
            )

        with st.expander(
            f"🔴 Customer {customer_id} — Risk Score: {row['risk_score']}/100 | {row['risk_level']}",
            expanded=True
        ):
            col_l, col_r = st.columns([1, 2])
            with col_l:
                st.markdown("**Detected Signals:**")
                for signal in signal_summary.split(';'):
                    st.markdown(f"- {signal.strip()}")
                st.markdown(f"**Customer Tier:** {row['customer_tier']}")
                st.markdown(f"**Risk Score:** {row['risk_score']}/100")
            with col_r:
                st.markdown(result['raw'])

    st.divider()

    # ── Portfolio-Level Rules ──────────────────────────────────────────────────
    st.subheader("📋 Portfolio-Level Monitoring Rules")
    with st.spinner("Generating portfolio monitoring rules..."):
        portfolio_rules = generate_portfolio_rules("\n".join(all_signals))
    st.markdown(portfolio_rules)

else:
    st.info("👈 Upload your transactions CSV and click **Run Analysis** to begin.")
    st.markdown("""
    **Expected CSV columns:**
    `record_id, customer_id, transaction_datetime, merchant_country, channel,
    merchant_category, MCC_code, amount, currency, authorisation_outcome,
    customer_tier, cross_border_flag, high_risk_merchant_flag, risk_flag`
    """)
```

---

## PHASE 6 — RUN AND TEST

### Step 11: Run the App

```bash
streamlit run app.py
```

Your browser will open at `http://localhost:8501`

### Step 12: Test With Sample Data

If you don't have the real CSV yet, create `data/sample.csv`:

```csv
record_id,customer_id,transaction_datetime,merchant_country,channel,merchant_category,MCC_code,amount,currency,authorisation_outcome,customer_tier,cross_border_flag,high_risk_merchant_flag,risk_flag
1,C001,2024-01-15 23:10:00,Russia,online,gambling,7995,9500,USD,approved,STANDARD,1,1,HIGH
2,C001,2024-01-15 23:25:00,Russia,online,gambling,7995,9200,USD,approved,STANDARD,1,1,HIGH
3,C001,2024-01-15 23:40:00,Myanmar,online,wire_transfer,4829,9800,USD,approved,STANDARD,1,1,HIGH
4,C002,2024-01-16 14:00:00,Singapore,pos,retail,5411,150,SGD,approved,PREMIUM,0,0,LOW
5,C003,2024-01-17 02:15:00,Iran,online,quasi_cash,6051,5000,USD,declined,STANDARD,1,1,HIGH
```

---

## PHASE 7 — COMMON ERRORS AND FIXES

| Error | Fix |
|---|---|
| `ModuleNotFoundError: groq` | Run `pip install groq` |
| `GROQ_API_KEY not found` | Check your `.env` file exists and is loaded |
| `FAISS index error` | Install `faiss-cpu` not `faiss-gpu` unless you have CUDA |
| `Sentence transformer download slow` | First run downloads ~80MB model — normal |
| `KeyError: 'cross_border_flag'` | The code handles missing columns; check your CSV headers |
| `Streamlit reloading on every change` | Normal in dev mode — use `--server.runOnSave false` to disable |

---

## PHASE 8 — PRESENTATION POLISH

### What to Highlight to Judges

1. **Hybrid architecture** — "We compute risk features in Python first, then send only flagged customers to the LLM. This mirrors real AML systems."

2. **Structuring detection** — "We specifically look for transactions between 85–99% of the $10,000 threshold, which is a classic money laundering technique."

3. **Explainability** — "Every risk score is decomposed into five sub-scores. Compliance teams can audit exactly why a customer was flagged."

4. **Delegated KYC framing** — "Our review questions are written specifically for the trust-but-verify context — we ask what the client's own controls found."

5. **Operationalisable rules** — "Rules are expressed as measurable thresholds that could be loaded directly into a monitoring system."

---

## QUICK REFERENCE — FILE CHECKLIST

```
✅ .env                    (GROQ_API_KEY)
✅ requirements.txt
✅ risk_engine.py
✅ rag_engine.py
✅ llm_engine.py
✅ app.py
✅ policy_docs/aml_policy.txt
✅ data/transactions.csv    (your actual dataset)
```

**Run order:**
1. `pip install -r requirements.txt`
2. Add GROQ_API_KEY to `.env`
3. Add your CSV to `data/`
4. `streamlit run app.py`
