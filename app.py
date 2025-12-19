# app.py
"""
LoanScore+ — Streamlit app (JSON input)
Cloud-ready version with Groq LLM
"""

import os
import json
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from src.feature_extraction import extract_features_from_json

# ---------------- ENV CHECK ----------------
st.write("Groq key loaded:", bool(os.environ.get("GROQ_API_KEY")))

st.set_page_config(page_title="LoanScore+ (JSON) — Professional", layout="wide")
st.title("LoanScore+ — Private Demo (JSON Upload)")

# ---------------- LOAD MODEL ----------------
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

# ---------------- EMI ----------------
def compute_emi_affordability(user_json):
    income = user_json.get("user_profile", {}).get("income_monthly")
    loans = user_json.get("loans", [])
    current_emi = sum(l.get("monthly_emi", 0) for l in loans)

    if income:
        return {
            "income_monthly": income,
            "current_emi_load": current_emi,
            "safe_min": round(income * 0.20, 2),
            "safe_max": round(income * 0.40, 2),
            "utilization_pct": round((current_emi / income) * 100, 2)
        }
    return {}

# ---------------- CREDIT FLAGS ----------------
def compute_credit_flags(features):
    flags = []
    flags.append(("Bounce events", "No bounces — healthy behaviour", "green"))
    flags.append(("Repayment rate", "Good repayment consistency", "green"))
    flags.append(("Spending volatility", "Moderate volatility", "yellow"))
    return flags

# ---------------- FEATURE IMPORTANCE ----------------
def get_top_features(model, X, names, k=5):
    try:
        imp = model.feature_importances_
        idx = np.argsort(-imp)[:k]
        return [(names[i], float(imp[i]), float(X.iloc[0, i])) for i in idx]
    except:
        return []

# ---------------- FILE UPLOAD ----------------
uploaded = st.file_uploader("Upload user JSON", type=["json"])
if not uploaded:
    st.stop()

user_json = json.load(uploaded)
st.subheader("User Profile")
st.json(user_json.get("user_profile", {}))

# ---------------- FEATURES ----------------
feats = extract_features_from_json(user_json)
X = pd.DataFrame([feats])[model_features].fillna(0)

# ---------------- SCORE ----------------
score = round(float(model.predict(X)[0]), 2)
st.metric("Eligibility Score (0–100)", score)

# ---------------- TOP FACTORS ----------------
top_feats = get_top_features(model, X, model_features)

top_feat_txt = "\n".join(
    [f"- {n}: value={round(v,2)}, impact={round(c,3)}"
     for n, c, v in top_feats]
)

st.subheader("Top Contributing Factors")
for n, c, v in top_feats:
    st.write(f"**{n}** → `{round(v,2)}`")

# ---------------- EMI ----------------
emi = compute_emi_affordability(user_json)

emi_txt = (
    f"Income: {emi.get('income_monthly')}\n"
    f"Current EMI: {emi.get('current_emi_load')}\n"
    f"Safe EMI range: {emi.get('safe_min')} – {emi.get('safe_max')}\n"
    f"Utilization%: {emi.get('utilization_pct')}"
)

st.subheader("EMI Affordability")
st.write(emi)

# ---------------- FLAGS ----------------
flags = compute_credit_flags(feats)
flags_txt = "\n".join([f"- {t}: {m}" for t, m, _ in flags])

# ---------------- GROQ LLM ----------------
st.subheader("AI-Generated Loan Explanation")

try:
    from src.llm_cloud import run_groq_llm

    llm_prompt = f"""
Loan Eligibility Score: {score}

Top contributing factors:
{top_feat_txt}

EMI affordability:
{emi_txt}

Credit behaviour flags:
{flags_txt}

Write:
• A short explanation suitable for a loan applicant
• 3 actionable tips to improve eligibility
"""

    with st.spinner("Generating AI explanation..."):
        explanation = run_groq_llm(llm_prompt)

    st.markdown("### AI Explanation")
    st.markdown(explanation)

except Exception as e:
    st.warning("AI explanation unavailable")
    st.code(str(e))

st.info("Demo uses synthetic data only.")
