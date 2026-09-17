"""TelcoCare AI: Streamlit front-end for retention agents.

Run (with the API already running):
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="TelcoCare AI", page_icon="📡", layout="wide")


def api(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=300, **kwargs)
    except requests.ConnectionError:
        st.error(f"Cannot reach the API at {API_URL}. Start it with `uvicorn api.main:app`.")
        st.stop()
    if not r.ok:
        detail = r.json().get("detail", r.text) if r.headers.get("content-type", "").startswith("application/json") else r.text
        raise RuntimeError(detail)
    return r.json()


RISK_BADGE = {"high": "🔴 High", "medium": "🟠 Medium", "low": "🟢 Low"}

# --- Sidebar -------------------------------------------------------------------
with st.sidebar:
    st.title("📡 TelcoCare AI")
    st.caption("Churn prediction + retention assistant")

    health = api("GET", "/health")
    llm = health["llm"]
    st.markdown(
        f"**Model:** {'✅ ' + health['model'].get('type', '') if health['model']['loaded'] else '❌ not trained'}  \n"
        f"**LLM:** {'✅' if llm['model_available'] else ('⚠️ model missing' if llm['reachable'] else '❌ Ollama offline')} `{llm['model']}`"
    )

    ids = api("GET", "/customers", params={"n": 30})["customer_ids"]
    manual = st.text_input("Customer ID", placeholder="e.g. 7590-VHVEG")
    customer_id = manual.strip() or st.selectbox("…or pick a sample customer", ids)

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []

left, right = st.columns([2, 3], gap="large")

# --- Customer panel ------------------------------------------------------------
with left:
    st.subheader(f"Customer {customer_id}")
    try:
        pred = api("GET", f"/predict/{customer_id}")
        profile = api("GET", f"/customers/{customer_id}")
    except RuntimeError as e:
        st.warning(str(e))
        st.stop()

    c1, c2, c3 = st.columns(3)
    c1.metric("Churn probability", f"{pred['churn_probability']:.0%}")
    c2.metric("Risk", RISK_BADGE[pred["risk_level"]])
    c3.metric("Tenure", f"{profile['tenure']} mo")

    st.markdown("**Top risk drivers**")
    drivers = pd.DataFrame(pred["top_reasons"])
    drivers["label"] = drivers["feature"] + " = " + drivers["value"]
    st.bar_chart(drivers.set_index("label")["impact"], horizontal=True, height=220)
    st.caption("Positive impact increases churn risk, negative impact decreases it (SHAP, log-odds).")

    with st.expander("Account details"):
        keys = ["Contract", "InternetService", "PaymentMethod", "MonthlyCharges", "TotalCharges",
                "OnlineSecurity", "TechSupport", "StreamingTV", "PaperlessBilling"]
        st.table(pd.DataFrame({"value": {k: str(profile[k]) for k in keys}}))

# --- Chat panel ----------------------------------------------------------------
with right:
    st.subheader("💬 Retention assistant")
    st.session_state.setdefault("messages", [])

    suggestions = [
        f"Why is customer {customer_id} at risk, and which campaign should I offer?",
        f"{customer_id} numaralı müşteri için bir arama senaryosu hazırla.",
        "List the 5 riskiest month-to-month customers.",
    ]
    cols = st.columns(len(suggestions))
    clicked = next((s for col, s in zip(cols, suggestions) if col.button(s, use_container_width=True)), None)

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            for t in m.get("trace", []):
                with st.expander(f"🔧 {t['name']}({', '.join(f'{k}={v!r}' for k, v in t['arguments'].items())}) · {t['duration_ms']} ms"):
                    st.json(t["result"])

    prompt = st.chat_input("Ask about this customer, campaigns, policies…") or clicked
    if prompt:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking…"):
            try:
                res = api("POST", "/chat", json={"message": prompt, "history": history[-10:]})
                st.session_state.messages.append(
                    {"role": "assistant", "content": res["answer"], "trace": res["trace"]}
                )
            except RuntimeError as e:
                st.session_state.messages.append({"role": "assistant", "content": f"⚠️ {e}"})
        st.rerun()
