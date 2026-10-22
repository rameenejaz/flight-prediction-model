# ✈️ Flight Ticket Price Prediction

**Live app:** https://flight-prediction-model-24jznugnvxzgr5mksxuxwf.streamlit.app

Predicts the ticket price of Indian domestic flights from the airline, route, departure and arrival time, stops, class,
journey duration and days before departure. Five regression algorithms were compared with **10-fold cross-validation**
on the full dataset of **300,153 flights**, and the best model is deployed as a Streamlit web app.

**Student:** Rameen Muzammil Ejaz (FA24-BSE-160)
**Course:** Machine Learning Fundamentals (AIC354), COMSATS University Islamabad, Lahore Campus
**Instructor:** Dr. Rao Muhammad Adeel Nawab

## Results (10-fold cross-validation)

| Rank | Model | Mean RMSE (₹) | MAE (₹) | R² |
|---|---|---|---|---|
| 1 | **XGBoost** (deployed) | **2,599** | 1,385 | **0.9869** |
| 2 | LightGBM | 2,633 | 1,409 | 0.9865 |
| 3 | Extra Trees | 2,664 | 1,210 | 0.9862 |
| 4 | AdaBoost | 4,430 | 2,682 | 0.9619 |
| 5 | Ridge Regression | 6,811 | 4,519 | 0.9100 |

## Files

| File | Purpose |
|---|---|
| `Flight_Price_Prediction_10Fold_CV.ipynb` | Google Colab notebook: data analysis, 10-fold CV of 5 models, charts, final model |
| `app.py` | Streamlit web app |
| `flight_price_model.joblib` | Trained model (preprocessing + XGBoost in one pipeline) |
| `model_meta.json` | Dropdown options and CV results used by the app |
| `requirements.txt` | Python libraries for the app (works on Python 3.12, 3.13 and 3.14) |
| `cv_fold_results.csv`, `cv_summary.csv` | Cross-validation results |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Dataset:** [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) (Kaggle, collected from EaseMyTrip).
# flight-prediction-model
