# 🌫️ Air Quality Analysis and Prediction Using Machine Learning

This project analyzes and predicts air quality in Texas (2022–2024) using historical EPA data. It explores pollutant trends, detects anomalies, classifies AQI into health bands, and forecasts future pollution levels using ML and time-series models.

---

## 📌 Objectives

- Predict AQI values using regression models
- Classify AQI into health-related categories
- Forecast future AQI trends using LSTM and ARIMA
- Detect anomalies in pollution using Isolation Forest
- Generate policy-relevant insights for better air quality management

---

## 🧰 Tech Stack

- **Language**: Python (Pandas, NumPy, Matplotlib, Scikit-learn, TensorFlow, Statsmodels)
- **Models**:
  - Regression: Linear Regression, Random Forest, XGBoost
  - Classification: Decision Tree, KNN, SVM, MLP (Neural Network)
  - Forecasting: LSTM, ARIMA
  - Anomaly Detection: Isolation Forest
- **Tools**: Jupyter Notebook, Excel, PowerPoint

---

## 📂 Dataset

- **Source**: [U.S. Environmental Protection Agency (EPA)](https://www.epa.gov/)
- **Size**: 140,000 rows × 21 columns
- **Coverage**: Texas (2022–2024)
- **Pollutants**: PM10, CO, NO₂, O₃, Pb
- **Target**: Daily AQI and AQI Category

---

## 🔍 EDA Highlights

- **Seasonal AQI Trends**: High in summer (O₃ spikes), lower in winter.
- **Top Pollutants**: Ozone (O₃) and PM10 have the strongest impact on AQI.
- **Geospatial Mapping**: Identified high-risk counties using lat-long scatterplots.
- **Anomalies**: Over 550 pollution anomalies detected (likely wildfires, leaks).

---

## ⚙️ Modeling Summary

### 1. 📉 AQI Prediction (Regression)

| Model             | RMSE | MAE  | R² Score |
|------------------|------|------|----------|
| Linear Regression| 19.34| 14.99| 0.04     |
| Random Forest     | 1.54 | 0.19 | **0.999** |
| XGBoost           | 1.50 | 0.20 | 0.994    |

✅ **Random Forest** gave the best performance.

---

### 2. 🧪 AQI Classification (WHO Bands)

| Model        | Precision | Recall | F1-Score |
|--------------|-----------|--------|----------|
| SVM          | 0.98      | 0.46   | 0.61     |
| Decision Tree| 1.00      | 1.00   | **1.00** |
| KNN          | 0.98      | 0.97   | 0.997    |
| Neural Net   | 0.92      | 0.93   | 0.98     |

✅ **Decision Tree and KNN** showed top accuracy.

---

### 3. ⏳ AQI Forecasting (Time Series)

- **LSTM**: RMSE = **5.57**, good for short-term trends
- **ARIMA**: Stable long-term forecast but less responsive to sudden spikes

✅ LSTM captured seasonality; ARIMA useful for trend stability.

---

## 🧪 Hypothesis Testing

- ✅ ML models significantly improve AQI predictions (p-value < 0.05)
- ✅ Classification models effectively categorize AQI health bands
- ✅ Time-series models (LSTM, ARIMA) can capture seasonal trends

---

## 📌 Insights & Policy Implications

- Pollution spikes occur in summer; policies should be **seasonal**
- **O₃ & PM10** are critical for mitigation
- Detected anomalies can pre-warn **wildfires** or **industrial leaks**
- Geospatial clustering supports **regional action planning**

---

## 🚫 Limitations

- Meteorological data (e.g., wind, humidity) was not included
- Some counties have fewer monitoring stations (regional bias)
- SO₂ and other pollutants were underrepresented

---

## 📁 Folder Structure

```bash
├── AirQuality_Dataset.xlsx
├── Air_Quality_Analysis_and_Prediction_Using_Machine_Learning.ipynb
└── README.md
