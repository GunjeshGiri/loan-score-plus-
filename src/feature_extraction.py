"""feature_extraction.py
Functions to extract ML-ready features from a single user JSON.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import random

def extract_features_from_json(user_json):
    txns = pd.DataFrame(user_json["transactions"])
    if txns.empty:
        raise ValueError("No transactions in JSON")
    txns['timestamp'] = pd.to_datetime(txns['timestamp'])
    total_txns_12m = len(txns)
    total_debits = txns[txns['type']=='debit']['amount'].sum()
    total_credits = txns[txns['type']=='credit']['amount'].sum()
    avg_txn_value_12m = txns['amount'].mean()
    ontime_payments_12m = int(txns[(txns['is_emi']==1) & (txns['emi_success']==1)].shape[0])
    late_payments_12m = int(txns[(txns['is_emi']==1) & (txns['emi_success']==0)].shape[0])
    repayment_rate_12m = ontime_payments_12m / (ontime_payments_12m + late_payments_12m + 1e-9)
    avg_balance_12m = txns['balance_after_txn'].mean()
    loans = user_json.get('loans', [])
    max_loan_taken = 0
    if loans:
        max_loan_taken = max([l.get('loan_amount',0) for l in loans])
    failed = txns[(txns['is_emi']==1) & (txns['emi_success']==0)]
    if failed.empty:
        months_since_last_default = 999
    else:
        last_failed = failed['timestamp'].max()
        months_since_last_default = max(0, (datetime.now() - last_failed).days // 30)
    app_counts = txns['upi_app'].value_counts().to_dict()
    loyalty_score = min(100, (app_counts.get('paytm',0)*0.4 + app_counts.get('gpay',0)*0.4 + app_counts.get('phonepe',0)*0.2) / max(1,total_txns_12m) * 100 + random.random()*10)
    days_active_12m = txns['timestamp'].dt.date.nunique()
    txns['month'] = txns['timestamp'].dt.to_period('M')
    monthly_totals = txns.groupby('month')['amount'].sum()
    monthly_volatility = monthly_totals.std() if len(monthly_totals)>1 else 0.0
    app_logins_monthly_avg = int(txns['month'].value_counts().mean())
    bounce_count = int(txns['is_bounce'].sum())
    refund_count = int(txns['is_refund'].sum())
    features = {
        "total_txns_12m": int(total_txns_12m),
        "avg_txn_value_12m": float(round(avg_txn_value_12m,2)),
        "ontime_payments_12m": int(ontime_payments_12m),
        "late_payments_12m": int(late_payments_12m),
        "repayment_rate_12m": float(round(repayment_rate_12m,4)),
        "avg_balance_12m": float(round(avg_balance_12m,2)),
        "max_loan_taken": int(max_loan_taken),
        "months_since_last_default": int(months_since_last_default),
        "loyalty_score": float(round(loyalty_score,2)),
        "days_active_12m": int(days_active_12m),
        "app_logins_monthly_avg": int(app_logins_monthly_avg),
        "monthly_volatility": float(round(monthly_volatility,2)),
        "bounce_count": int(bounce_count),
        "refund_count": int(refund_count)
    }
    return features
