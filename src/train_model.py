"""train_model.py
Reads json_users in data/json_users/, extracts features and trains a RandomForestRegressor.
Saves model to models/loan_score_model.pkl
"""
import os, glob, json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from src.feature_extraction import extract_features_from_json

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'json_users')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(OUT_DIR, exist_ok=True)

json_files = sorted(glob.glob(os.path.join(DATA_DIR, 'user_*.json')))
rows = []
for jf in json_files:
    with open(jf, 'r') as f:
        user_json = json.load(f)
    feats = extract_features_from_json(user_json)
    base = 30 + 40*feats['repayment_rate_12m'] + 0.0003*feats['avg_balance_12m'] + 0.1*feats['loyalty_score'] - 0.3*feats['late_payments_12m'] - 0.2*feats['bounce_count']
    noise = np.random.normal(0,7)
    feats['eligibility_score'] = float(np.clip(base + noise, 0, 100))
    feats['user_id'] = user_json['user_profile']['user_id']
    rows.append(feats)

df = pd.DataFrame(rows)
feature_cols = [c for c in df.columns if c not in ('user_id','eligibility_score')]
X = df[feature_cols].fillna(0)
y = df['eligibility_score']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)
print('MAE:', mae, 'R2:', r2)

joblib.dump({'model': model, 'features': feature_cols}, os.path.join(OUT_DIR, 'loan_score_model.pkl'))
df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'yearly_features_with_target.csv'), index=False)
print('Saved model and features.')
