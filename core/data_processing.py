def compute_emi_affordability(user_json, features):
    income = user_json.get('user_profile',{}).get('income_monthly')
    loans = user_json.get('loans', [])
    current_emi_load = sum([l.get('monthly_emi',0) for l in loans]) if loans else 0
    return {'income_monthly': income, 'current_emi_load': current_emi_load, 'safe_min': income*0.2 if income else None, 'safe_max': income*0.4 if income else None}

def compute_credit_flags(features):
    flags=[]
    if features.get('bounce_count',0)==0:
        flags.append(('Bounce events','No bounces','green'))
    else:
        flags.append(('Bounce events','Has bounces','red'))
    if features.get('repayment_rate_12m',1)>=0.95:
        flags.append(('Repayment rate','Excellent','green'))
    else:
        flags.append(('Repayment rate','Needs improvement','yellow'))
    return flags
