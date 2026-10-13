# Credit Card Default Risk Prediction System

An end-to-end machine learning system that predicts whether a credit card customer will default next month, and explains **why** using SHAP. Built with LightGBM, FastAPI and React.

## Problem Statement

Banks need to identify high-risk customers before they default. A model that only outputs "default / no default" is not enough for credit decisions. The reasoning behind each prediction has to be visible too. This project predicts default risk and returns the top factors driving each individual prediction.

## Dataset

- **UCI Default of Credit Card Clients** dataset
- 30,000 records, 23 raw features (credit limit, demographics, 6 months of payment status, bill amounts and payment amounts)
- Target: `default.payment.next.month`
- Class imbalance: ~78% non-default vs ~22% default
- No missing values

## ML Approach

1. **EDA**: checked missing values and class distribution.
2. **Imbalance handling**: used LightGBM's `scale_pos_weight` (~3.52) instead of SMOTE, so no synthetic data is created.
3. **Split**: 80/20 stratified train/test split.
4. **Hyperparameter tuning**: GridSearchCV (3-fold CV, F1 scoring) over `max_depth`, `n_estimators`, `learning_rate` and `num_leaves`.
   - Best params: `learning_rate=0.01`, `max_depth=8`, `n_estimators=200`, `num_leaves=40`
5. **Feature engineering**: 7 derived features: `avg_pay_delay`, `max_pay_delay`, `avg_bill_amt`, `utilization_ratio`, `total_bill_amt`, `total_pay_amt`, `payment_ratio`.
6. **Explainability**: SHAP `TreeExplainer` for global feature importance and per-customer explanations.

## Results (Test Set, 6,000 samples)

| Metric | Value |
|---|---|
| ROC-AUC | 0.77 |
| F1-score (default class) | 0.53 |
| Recall (default class) | 0.60 |
| Precision (default class) | 0.47 |
| Accuracy | 0.76 |

**Key findings**
- Recent payment behaviour (`PAY_0`, the latest month's repayment status) is the strongest predictor of default.
- The 7 engineered features gave only a marginal gain (ROC-AUC 0.767 to 0.771). Tree-based models already capture these interactions on their own.
- F1 and ROC-AUC are used instead of accuracy because accuracy is misleading on imbalanced data.

## Architecture

```
React Frontend  --->  FastAPI Backend  --->  LightGBM Model + SHAP Explainer
(form + results)      (/predict endpoint)    (loaded once at startup)
```

- **Frontend**: React form with 23 inputs, calls the API via Axios and displays the prediction, probability and top factors.
- **Backend**: FastAPI with a Pydantic-validated schema (23 raw fields). The 7 engineered features are computed server-side, so the client never sends them.
- **Model**: trained LightGBM classifier saved with joblib, loaded once at server startup.

## API

### `POST /predict`

**Request body** (23 fields): `LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2–PAY_6, BILL_AMT1–6, PAY_AMT1–6`

**Response**
```json
{
  "prediction": "No Default",
  "probability": 0.3507,
  "top_factors": [
    {"feature": "MARRIAGE", "impact": -0.7386},
    {"feature": "BILL_AMT1", "impact": 0.3338},
    {"feature": "max_pay_delay", "impact": -0.2724}
  ]
}
```

`impact` is the SHAP value (log-odds scale). Positive values increase default risk, negative values decrease it.

Interactive docs are available at `http://127.0.0.1:8000/docs` (Swagger UI).

## Project Structure

```
credit-risk-model/
├── backend/
│   ├── data/raw/          # UCI dataset
│   ├── models/            # model.pkl, feature_names.pkl
│   └── src/main.py        # FastAPI app
├── frontend/
│   └── src/
│       ├── api.js         # Axios API call
│       └── components/PredictionForm.jsx
└── eda.ipynb              # EDA, training, tuning, SHAP
```

## How to Run Locally

**Backend**
```bash
pip install pandas numpy scikit-learn lightgbm shap fastapi uvicorn joblib
cd backend/src
uvicorn main:app --reload
```

**Frontend** (in a second terminal)
```bash
cd frontend
npm install
npm start
```

Open `http://localhost:3000`.

## Tech Stack

- **ML**: Python, Pandas, NumPy, scikit-learn, LightGBM, SHAP
- **Backend**: FastAPI, Pydantic, Uvicorn
- **Frontend**: React, Axios
- **Tools**: Git, GitHub, Jupyter

## Roadmap

- [ ] Docker containerization (backend and frontend)
- [ ] Deployment
- [ ] Threshold tuning to improve recall on defaulters
- [ ] SHAP chart on the frontend
- [ ] JWT authentication

## Author

**Vishnu Kant**: [GitHub](https://github.com/Vishnkant790)
