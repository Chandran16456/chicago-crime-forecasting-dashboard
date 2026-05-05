# CrimeCast Chicago: Crime Forecasting & Risk Intelligence Dashboard

CrimeCast Chicago is an end-to-end crime forecasting and risk analytics dashboard built using historical Chicago crime records. The project processes large-scale raw crime data, builds time-series forecasting models, generates district and crime-type risk predictions, and presents the results through a FastAPI-powered web dashboard with an interactive Chicago map.

---

## Project Overview

This project analyzes more than 20 years of Chicago crime data to forecast future crime volume and identify risk patterns across districts, crime categories, and time periods.

The system predicts:

- Next 7-day citywide crime count
- Next 30-day citywide crime count
- District-level crime risk
- Crime-type-level risk
- High-risk time periods by district, crime type, day, and hour
- Map-based district risk insights

The goal is not to predict individual crimes or people. The system forecasts aggregated crime volume and relative risk patterns using historical data.

---

## Key Features

### Forecasting

- Next 7-day forecast
- Next 30-day forecast
- Citywide crime volume prediction
- Baseline, ARIMA, SARIMA, and XGBoost model comparison

### District Risk Analytics

- District-level risk ranking
- Predicted 30-day crime count by district
- Risk percentage compared with recent baseline
- Interactive district detail pop-ups

### Crime Type Risk

- Crime-type ranking
- Predicted 30-day crime count by category
- Risk percentage by crime type
- Low-volume confidence handling

### High-Risk Time Period Detection

- Risk by district
- Risk by crime type
- Risk by day of week
- Risk by hour of day
- Pop-up detail panels for deeper analysis

### Interactive Chicago Map

- District-level map markers
- Clickable prediction pop-ups
- District filters
- Crime-type filters
- Map hide/show control for dashboard usability

---

## Tech Stack

### Data Engineering

- Python
- DuckDB
- Pandas
- NumPy
- PyArrow
- Parquet

### Machine Learning

- Baseline forecasting
- ARIMA
- SARIMA
- XGBoost
- Scikit-learn
- Statsmodels
- Joblib

### Backend

- FastAPI
- Uvicorn
- Pandas API data serving

### Frontend

- HTML
- CSS
- JavaScript
- Leaflet.js
- Chart.js

### Version Control

- Git
- GitHub

---

## Project Architecture

```mermaid
flowchart TD
    A[Raw Chicago Crime TXT File] --> B[DuckDB Data Loading]
    B --> C[Cleaned Crime Table]
    C --> D[Daily Aggregated Tables]
    D --> E[Feature Engineering]
    E --> F[Forecasting Models]
    F --> G[Model Evaluation]
    G --> H[Dashboard API Files]
    H --> I[FastAPI Backend]
    I --> J[HTML CSS JS Frontend]
    J --> K[Interactive Dashboard]
