import joblib
import pandas as pd
from src.feature_extraction import extract_features_from_json

MODEL_PATH = 'models/loan_score_model.pkl'

def load_model_and_features():
    data = joblib.load(MODEL_PATH)
    return data['model'], data['features']

def predict_for_user(model, feature_cols, user_json):
    feats = extract_features_from_json(user_json)
    X = pd.DataFrame([feats])[feature_cols].fillna(0)
    score = float(model.predict(X)[0])
    return {'score': round(score,2), 'features': feats, 'X': X}
