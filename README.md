# Food Delivery ETA Prediction

## Problem
Predict delivery time based on order, traffic, and weather conditions.

# Exploratory Data Analysis

## Objective
The goal of this project is to analyze factors affecting food delivery time and build a predictive model for estimating delivery duration.

## Key Questions
- How does distance affect delivery time?
- What is the impact of traffic and weather?
- Are there peak hours with higher delays?
- Do different cities show different patterns?

## OUTCOMES OF EDA
- Distance shows strong positive correlation with delivery time
- High traffic increases ETA by ~40%
- Stormy weather causes maximum delays
- Peak hours (12–2 PM, 7–9 PM) show higher delivery times


### Key Results:
- **Winning Model:** Linear Regression (surpassing XGBoost/Random Forest due to high-quality feature engineering).
- **Accuracy:** MAE of **4.07 minutes**.
- **Variance Explained (R^2): **95.31%**.

---

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, Scikit-Learn, XGBoost, Seaborn
- **Deployment:** Streamlit (Web UI)
- **Serialization:** Joblib

---

## 🧠 Feature Engineering
To achieve high accuracy with a simple model, I engineered several high-impact features:
1. **Interaction Terms:** `distance * traffic_level` (captures how traffic slows down longer routes non-linearly).
2. **Temporal Features:** Extracted `hour` and `day_of_week` to identify lunch/dinner peaks.
3. **Peak Hour Indicator:** Boolean flag for high-demand windows (12-2 PM, 7-9 PM).
4. **Outlier Removal:** Applied the IQR method to remove delivery "anomalies" (e.g., cancelled or extremely delayed orders).

---

## How to Run

## 1. Install Dependencies
pip install -r requirements.txt

## Run Instructions

### Train and Evaluate
python src/main.py

### Run API
streamlit run app.py