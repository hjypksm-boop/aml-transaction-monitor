import os

import streamlit as st
import plotly.express as px
from risk_engine import (
    load_and_prepare, compute_customer_features,
    compute_risk_score, get_top_flagged, build_customer_signal_summary
)
from rag_engine import build_vector_store, retrieve_policy_context
from llm_engine import (
    generate_customer_narrative, generate_portfolio_rules, get_api_key, get_model
)

SAMPLE_DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_transactions.csv")

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nium AML Monitor",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Nium Transaction Monitoring — Trust but Verify")
st.caption("UC1 | AML Risk Intelligence Dashboard")


# ─── Cached resources ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building policy knowledge base (first run downloads the embedding model)...")
def get_vector_store():
    return build_vector_store()


@st.cache_data(show_spinner=False)
def cached_narrative(customer_id, risk_score, risk_level, signal_summary, policy_context, customer_tier, model):
    # `model` is part of the cache key so switching GROQ_MODEL regenerates output.
    return generate_customer_narrative(
        customer_id=customer_id,
        risk_score=risk_score,
        risk_level=risk_level,
        signal_summary=signal_summary,
        policy_context=policy_context,
        customer_tier=customer_tier,
    )


@st.cache_data(show_spinner=False)
def cached_portfolio_rules(signals, model):
    return generate_portfolio_rules(signals)


# ─── Sidebar: Data input ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Input")
    use_sample = st.toggle("Use sample data", value=False,
                           help="Analyse the bundled synthetic dataset (data/sample_transactions.csv).")
    uploaded_file = None
    if not use_sample:
        uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])
    run_button = st.button("▶ Run Analysis", type="primary")
    st.divider()
    if get_api_key():
        st.caption(f"LLM: Groq `{get_model()}`")
    else:
        st.warning("GROQ_API_KEY not found — risk scoring will run, but AI narratives are disabled.")
    st.caption("Powered by Python Risk Engine + Groq + FAISS RAG")

source = SAMPLE_DATA_PATH if use_sample else uploaded_file
source_key = "sample" if use_sample else (uploaded_file.file_id if uploaded_file else None)

# Keep results on screen across reruns (e.g. expanding a table) until the input changes.
if run_button and source is not None:
    st.session_state["analysed_source"] = source_key
if st.session_state.get("analysed_source") != source_key:
    st.session_state.pop("analysed_source", None)

# ─── Main Logic ────────────────────────────────────────────────────────────────
if source is not None and st.session_state.get("analysed_source") == source_key:

    with st.spinner("Loading transactions..."):
        if hasattr(source, "seek"):
            source.seek(0)
        try:
            df = load_and_prepare(source)
        except ValueError as e:
            st.error(str(e))
            st.stop()

    with st.spinner("Computing behavioural risk features..."):
        features = compute_customer_features(df)
        scored = compute_risk_score(features)
        top_flagged = get_top_flagged(scored, n=5)

    vectorstore = get_vector_store()

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    st.subheader("Portfolio Overview")
    st.caption(f"{len(df):,} transactions · {len(scored)} customers"
               + (" · sample data" if use_sample else ""))
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
                     color_discrete_map={'HIGH': '#e74c3c', 'MEDIUM': '#f39c12', 'LOW': '#2ecc71'})
        st.plotly_chart(fig, width="stretch")

    with col_b:
        st.subheader("Top 10 Customers by Risk Score")
        top10 = scored.head(10)[['customer_id', 'risk_score', 'risk_level',
                                 'max_velocity_per_hour', 'risky_mcc_count',
                                 'cross_border_count', 'structuring_count']]
        st.dataframe(top10, width="stretch", hide_index=True)

    st.divider()

    # ── Full Customer Table ────────────────────────────────────────────────────
    with st.expander("📋 Full Customer Risk Table"):
        display_cols = ['customer_id', 'risk_score', 'risk_level', 'total_transactions',
                        'avg_tx_size', 'max_velocity_per_hour', 'structuring_count',
                        'risky_mcc_count', 'cross_border_count', 'late_night_ratio',
                        'high_risk_country_count', 'declined_ratio', 'customer_tier']
        st.dataframe(scored[display_cols], width="stretch", hide_index=True)

    st.divider()

    # ── Top Flagged Customers — AI Narratives ──────────────────────────────────
    st.subheader("🚨 Top Flagged Customers — AI Compliance Narratives")

    llm_enabled = bool(get_api_key())
    model = get_model()
    all_signals = []

    if top_flagged.empty:
        st.success("No customers reached the HIGH risk threshold.")

    for _, row in top_flagged.iterrows():
        customer_id = str(row['customer_id'])
        signal_summary = build_customer_signal_summary(row)
        all_signals.append(f"Customer {customer_id}: {signal_summary}")

        policy_context = retrieve_policy_context(vectorstore, query=signal_summary)

        result = None
        error = None
        if llm_enabled:
            with st.spinner(f"Generating narrative for Customer {customer_id}..."):
                try:
                    result = cached_narrative(
                        customer_id, float(row['risk_score']), str(row['risk_level']),
                        signal_summary, policy_context, str(row['customer_tier']), model,
                    )
                except Exception as e:
                    error = e

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
                if result:
                    st.markdown(result['raw'])
                elif error:
                    st.error(f"Groq request failed: {error}")
                else:
                    st.info("Add GROQ_API_KEY to generate an AI narrative. Retrieved policy context:")
                    st.text(policy_context)

    # ── Portfolio-Level Rules ──────────────────────────────────────────────────
    if llm_enabled and all_signals:
        st.divider()
        st.subheader("📋 Portfolio-Level Monitoring Rules")
        with st.spinner("Generating portfolio monitoring rules..."):
            try:
                st.markdown(cached_portfolio_rules("\n".join(all_signals), model))
            except Exception as e:
                st.error(f"Groq request failed: {e}")

else:
    st.info("👈 Upload your transactions CSV (or toggle **Use sample data**) and click **Run Analysis** to begin.")
    st.markdown("""
    **Expected CSV columns:**
    `record_id, customer_id, transaction_datetime, merchant_country, channel,
    merchant_category, MCC_code, amount, currency, authorisation_outcome,
    customer_tier, cross_border_flag, high_risk_merchant_flag, risk_flag`
    """)
