# LoanScore+ (JSON) — Project
This repository contains a synthetic LoanScore system that computes an eligibility score from a user's UPI-style JSON transaction history.

## Structure
- `data/json_users.zip` — synthetic dataset (100 users). Extract into `data/json_users/`.
- `data/yearly_features_with_target.csv` — aggregated features with synthetic target.
- `models/loan_score_model.pkl` — trained RandomForest model (if present).
- `src/` — helper scripts:
  - `build_dataset.py` — regenerate synthetic JSON dataset.
  - `feature_extraction.py` — extract ML features from a user JSON.
  - `train_model.py` — train model from JSON files and save model.
- `app.py` — Streamlit app (upload JSON -> predict -> SHAP -> LLM stub)
- `requirements.txt` — Python dependencies

## Quick start
1. (Optional) Create a venv and install requirements: `pip install -r requirements.txt`
2. Extract sample data: `unzip data/json_users.zip -d data/json_users`
3. Train model: `python src/train_model.py`
4. Run Streamlit: `streamlit run app.py`
5. In the app upload a `user_###.json` from `data/json_users/`

## Notes
- All data is synthetic for academic/testing purposes.
- Do NOT upload real PII to public deployments.
- The LLM explanation is a stub. Replace with your API call and prompt template.
