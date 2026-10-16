import os
from functools import lru_cache

from groq import Groq
from dotenv import load_dotenv

# llama-3.3-70b-versatile was shut down for free/developer tiers on 2026-08-16;
# Groq recommends openai/gpt-oss-120b as the replacement. Override with GROQ_MODEL.
DEFAULT_MODEL = "openai/gpt-oss-120b"


def _get_setting(name: str, default: str | None = None) -> str | None:
    """Read a setting from st.secrets first (Streamlit Cloud), then .env / environment."""
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        # No secrets.toml present, or not running under Streamlit.
        pass
    load_dotenv()
    return os.getenv(name, default)


def get_api_key() -> str | None:
    return _get_setting("GROQ_API_KEY")


def get_model() -> str:
    return _get_setting("GROQ_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=1)
def _client(api_key: str) -> Groq:
    return Groq(api_key=api_key)


def _complete(prompt: str, max_tokens: int) -> str:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .streamlit/secrets.toml or .env.")

    model = get_model()
    kwargs = {}
    if model.startswith("openai/gpt-oss"):
        # Reasoning models spend completion tokens on hidden reasoning; keep it light
        # and leave headroom so the visible answer isn't truncated.
        kwargs["reasoning_effort"] = "low"
        max_tokens += 1500

    response = _client(api_key).chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_completion_tokens=max_tokens,
        **kwargs,
    )
    return response.choices[0].message.content or ""


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

    return {"raw": _complete(prompt, max_tokens=800)}


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

    return _complete(prompt, max_tokens=600)
