# app.py
"""
LoanScore+ — Streamlit app (JSON input)
Features:
- Load trained model from models/loan_score_model.pkl
- Upload a single user JSON file
- Extract features and predict loan eligibility score
- SHAP-based top contributors (if available)
- EMI affordability analysis
- Credit behavior flags
- Professional LLM explanation using LOCAL AI (Llama 3.2 1B)
- 100% offline, free, no API key required
"""

import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from src.feature_extraction import extract_features_from_json
from ui.layout import render_page

st.set_page_config(page_title="LoanScore+ (JSON) — Professional", layout="wide")
st.title("LoanScore+ — Private Demo (JSON Upload)")

# -----------------------------------------------------
# Load Model
# -----------------------------------------------------
def load_model():
    path = os.path.join("models", "loan_score_model.pkl")
    if not os.path.exists(path):
        return None, None
    data = joblib.load(path)
    return data.get("model"), data.get("features")

model, model_features = load_model()
if model is None:
    st.error("Model missing. Run: python src/train_model.py")
    st.stop()

# -----------------------------------------------------
# EMI Affordability
# -----------------------------------------------------
def compute_emi_affordability(user_json, features):
    income = user_json.get("user_profile", {}).get("income_monthly", None)
    loans = user_json.get("loans", [])
    current_emi_load = sum(l.get("monthly_emi", 0) for l in loans) if loans else 0.0

    txns = pd.DataFrame(user_json.get("transactions", []))
    if not txns.empty:
        emi_txns = txns[(txns["is_emi"] == 1) | (txns["merchant_category"] == "emi_payment")]
        if len(emi_txns) > 0:
            monthly_estimated = emi_txns["amount"].sum() / max(1, emi_txns["timestamp"].nunique())
            if current_emi_load == 0:
                current_emi_load = monthly_estimated

    if income:
        safe_min = round(income * 0.20, 2)
        safe_max = round(income * 0.40, 2)
        utilization_pct = round((current_emi_load / income) * 100, 2)
    else:
        safe_min = safe_max = utilization_pct = None

    return {
        "income_monthly": income,
        "current_emi_load": round(current_emi_load, 2),
        "safe_min": safe_min,
        "safe_max": safe_max,
        "utilization_pct": utilization_pct,
    }

# -----------------------------------------------------
# Credit Behaviour Flags
# -----------------------------------------------------
def compute_credit_flags(features):
    flags = []

    # Bounce count
    b = features.get("bounce_count", 0)
    if b == 0:
        flags.append(("Bounce events", "No bounces — healthy behaviour", "green"))
    elif b < 3:
        flags.append(("Bounce events", "Minor bounces — monitor", "yellow"))
    else:
        flags.append(("Bounce events", "Frequent bounces — high risk", "red"))

    # Repayment rate
    rr = features.get("repayment_rate_12m", 1.0)
    if rr >= 0.95:
        flags.append(("Repayment rate", "Excellent repayment consistency", "green"))
    elif rr >= 0.8:
        flags.append(("Repayment rate", "Needs improvement", "yellow"))
    else:
        flags.append(("Repayment rate", "Low repayment rate — risky", "red"))

    # Monthly volatility
    vol = features.get("monthly_volatility", 0)
    if vol < 5000:
        flags.append(("Spending volatility", "Stable spending", "green"))
    elif vol < 20000:
        flags.append(("Spending volatility", "Moderate volatility", "yellow"))
    else:
        flags.append(("Spending volatility", "High volatility — risky", "red"))

    # Late EMIs
    late = features.get("late_payments_12m", 0)
    if late == 0:
        flags.append(("Late EMIs", "No late payments", "green"))
    elif late <= 2:
        flags.append(("Late EMIs", "Occasional delays", "yellow"))
    else:
        flags.append(("Late EMIs", "Multiple missed EMIs — critical", "red"))

    return flags

# -----------------------------------------------------
# SHAP / Feature Importance
# -----------------------------------------------------
def get_top_features(model, X, feature_names, top_k=5):
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)[0]
        idx = np.argsort(-np.abs(values))[:top_k]
        return [(feature_names[i], float(values[i]), float(X.iloc[0, i])) for i in idx]
    except:
        try:
            imp = model.feature_importances_
            idx = np.argsort(-imp)[:top_k]
            return [(feature_names[i], float(imp[i]), float(X.iloc[0, i])) for i in idx]
        except:
            vals = np.abs(X.iloc[0])
            idx = np.argsort(-vals)[:top_k]
            return [(feature_names[i], float(vals[i]), float(X.iloc[0, i])) for i in idx]

# -----------------------------------------------------
# Upload JSON
# -----------------------------------------------------
uploaded = st.file_uploader("Upload user JSON (e.g. user_000.json)", type=["json"])
if not uploaded:
    st.info("Upload a JSON file to continue.")
    st.stop()

user_json = json.load(uploaded)

st.subheader("User Profile")
st.json(user_json.get("user_profile", {}))

# -----------------------------------------------------
# Feature Extraction
# -----------------------------------------------------
with st.spinner("Extracting features..."):
    feats = extract_features_from_json(user_json)

st.subheader("Extracted Features")
st.table(pd.DataFrame([feats]).T.rename(columns={0: "value"}))

# -----------------------------------------------------
# Score Prediction
# -----------------------------------------------------
X = pd.DataFrame([feats])[model_features].fillna(0)
score = round(float(model.predict(X)[0]), 2)
st.metric("Eligibility Score (0–100)", score)

# -----------------------------------------------------
# Top Contributors
# -----------------------------------------------------
st.subheader("Top Contributing Factors")
top_feats = get_top_features(model, X, model_features)
for n, c, v in top_feats:
    st.write(f"**{n}** → value `{round(v,2)}`, impact `{round(c,3)}`")

# -----------------------------------------------------
# EMI Affordability
# -----------------------------------------------------
st.subheader("EMI Affordability")
emi = compute_emi_affordability(user_json, feats)
st.write(emi)

# -----------------------------------------------------
# Credit Flags
# -----------------------------------------------------
st.subheader("Credit Behaviour Flags")
flags = compute_credit_flags(feats)
cols = st.columns(3)
for i, (title, msg, color) in enumerate(flags):
    c = cols[i % 3]
    if color == "green":
        c.success(f"{title}: {msg}")
    elif color == "yellow":
        c.warning(f"{title}: {msg}")
    else:
        c.error(f"{title}: {msg}")

# -----------------------------------------------------
# Local AI Explanation (Llama 3.2 1B)
# -----------------------------------------------------
st.subheader("LLM-Generated Insights by LoanScore+")

from llama_cpp import Llama

MODEL_PATH = "models/Llama-3.2-1B-Instruct-Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    st.error("Local model not found. Add Llama-3.2-1B-Instruct-Q4_K_M.gguf to models/ folder.")
else:
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=4096,
        n_threads=4,      # adjust for your CPU
        n_gpu_layers=0    # CPU-only mode
    )

    # Build the prompt
    system_prompt = (
        "You are a senior financial credit analyst for a digital lender. "
        "Write clear, professional, helpful explanations for customers. "
        "Avoid technical terms and machine learning references."
    )

    top_feat_txt = "\n".join([f"- {n}: value={round(v,2)}, impact={round(c,3)}" for n, c, v in top_feats])
    emi_txt = (
        f"Income: {emi['income_monthly']}\n"
        f"Current EMI: {emi['current_emi_load']}\n"
        f"Safe EMI range: {emi['safe_min']} – {emi['safe_max']}\n"
        f"Utilization%: {emi['utilization_pct']}"
    )
    flags_txt = "\n".join([f"- {t}: {msg}" for t, msg, _ in flags])

    user_prompt = f"""
Loan Eligibility Score: {score}

Key financial indicators:
{top_feat_txt}

EMI affordability:
{emi_txt}

Credit behaviour flags:
{flags_txt}

Write:
• A 3–5 sentence explanation suitable for a loan applicant.
• 3 practical steps to improve eligibility.
"""

    with st.spinner("Generating explanation using LLM..."):
        result = llm(
            f"<s>[INST]<<SYS>> {system_prompt} <</SYS>> {user_prompt} [/INST]",
            max_tokens=300,
            temperature=0.3,
        )

    response_text = result["choices"][0]["text"]
    st.markdown("### LLM Explanation")
    st.markdown(response_text)

st.markdown("---")
st.info("The information shown here is based on synthetic financial activity "
        "created for testing and validation. This environment does not process "
        "or store any personal customer data.")

from ui.layout import render_page
def main():

    mode = st.sidebar.radio("Mode", ["Single User", "Batch Processing"])

    if mode == "Single User":
        render_page()


if __name__ == "__main__":
    main()
