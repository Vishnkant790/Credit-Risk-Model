from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load('../models/model.pkl')
feature_names = joblib.load('../models/feature_names.pkl')

import shap
explainer = shap.TreeExplainer(model)


class CustomerData(BaseModel):
    LIMIT_BAL: float
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float

@app.post("/predict")
def predict(customer: CustomerData):
    data = customer.dict()

    pay_cols = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
    pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

    data['avg_pay_delay'] = sum(data[c] for c in pay_cols)/len(pay_cols)
    data['max_pay_delay'] = max(data[c] for c in pay_cols)
    data['avg_bill_amt'] = sum(data[c] for c in bill_cols) / len(bill_cols)
    data['utilization_ratio'] = data['avg_bill_amt'] / (data['LIMIT_BAL'] + 1)
    data['total_bill_amt'] = sum(data[c] for c in bill_cols)
    data['total_pay_amt'] = sum(data[c] for c in pay_amt_cols)
    data['payment_ratio'] = data['total_pay_amt'] / (data['total_bill_amt'] + 1)

    input_df = pd.DataFrame([data])
    input_df = input_df[feature_names]


    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[:, 1][0]
    shap_values = explainer.shap_values(input_df)
    feature_impact = dict(zip(feature_names, shap_values[0]))
    top_features = sorted(feature_impact.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    explanation = [
        {"feature": f, "impact": round(float(v), 4)} for f, v in top_features
    ]




    return {
        "prediction": "Default" if prediction == 1 else "No Default",
        "probability": round(float(probability), 4),
        "top_factors": explanation

    }