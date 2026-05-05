from pathlib import Path
import duckdb
import pandas as pd
import json

ROOT_DIR = Path(__file__).resolve().parents[1]

DB_PATH = ROOT_DIR / "db" / "chicago_crime.duckdb"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DASHBOARD_DIR = PROCESSED_DIR / "dashboard"

DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("BUILDING DASHBOARD-READY OUTPUTS")
print("=" * 80)

if not DB_PATH.exists():
    raise FileNotFoundError(f"DuckDB database not found: {DB_PATH}")

con = duckdb.connect(str(DB_PATH))

# --------------------------------------------------
# 1. Load model comparison
# --------------------------------------------------

print("\n[1/7] Loading model comparison")

model_comparison_path = PROCESSED_DIR / "model_comparison_current.csv"

if not model_comparison_path.exists():
    raise FileNotFoundError(
        "model_comparison_current.csv not found. "
        "Run baseline, ARIMA, and SARIMA scripts first."
    )

model_comparison = pd.read_csv(model_comparison_path)
model_comparison = model_comparison.sort_values("mae").reset_index(drop=True)

best_citywide_model = model_comparison.iloc[0]["model"]
best_citywide_group = model_comparison.iloc[0]["model_group"]

print("Best citywide model:", best_citywide_model)
print("Best citywide group:", best_citywide_group)

# --------------------------------------------------
# 2. Select best citywide 30-day forecast
# --------------------------------------------------

print("\n[2/7] Selecting best citywide forecast")

forecast_file_map = {
    "Baseline": "citywide_30day_forecast_baseline.csv",
    "ARIMA": "citywide_30day_forecast_arima.csv",
    "SARIMA": "citywide_30day_forecast_sarima.csv",
}

best_forecast_file = forecast_file_map.get(best_citywide_group)

if best_forecast_file is None:
    raise RuntimeError(f"No forecast file mapped for model group: {best_citywide_group}")

best_forecast_path = PROCESSED_DIR / best_forecast_file

if not best_forecast_path.exists():
    raise FileNotFoundError(f"Forecast file not found: {best_forecast_path}")

citywide_forecast = pd.read_csv(best_forecast_path)

citywide_forecast["date"] = pd.to_datetime(citywide_forecast["date"]).dt.strftime("%Y-%m-%d")
citywide_forecast["selected_model"] = best_citywide_model
citywide_forecast["model_group"] = best_citywide_group

citywide_7day = citywide_forecast.head(7).copy()
citywide_30day = citywide_forecast.copy()

citywide_7day_path = DASHBOARD_DIR / "api_citywide_7day_forecast.csv"
citywide_30day_path = DASHBOARD_DIR / "api_citywide_30day_forecast.csv"

citywide_7day.to_csv(citywide_7day_path, index=False)
citywide_30day.to_csv(citywide_30day_path, index=False)

print(f"Saved: {citywide_7day_path}")
print(f"Saved: {citywide_30day_path}")

# --------------------------------------------------
# 3. Build district risk map data
# --------------------------------------------------

print("\n[3/7] Building district risk map data")

district_map_path = PROCESSED_DIR / "district_map_points.csv"

if not district_map_path.exists():
    raise FileNotFoundError(
        "district_map_points.csv not found. "
        "Run scripts\\08_train_xgboost.py first."
    )

district_map = pd.read_csv(district_map_path)

# Cleaner risk explanation
district_map["risk_direction"] = district_map["risk_percent_vs_recent_baseline"].apply(
    lambda x: "Above Recent Baseline" if x > 0 else "Below Recent Baseline"
)

district_map["display_risk_percent"] = district_map["risk_percent_vs_recent_baseline"].round(2)

district_map = district_map[
    [
        "district",
        "predicted_30_day_crime_count",
        "expected_30_day_count",
        "display_risk_percent",
        "risk_direction",
        "risk_level",
        "latitude",
        "longitude",
    ]
].copy()

district_map = district_map.sort_values(
    "display_risk_percent",
    ascending=False
).reset_index(drop=True)

district_map_output = DASHBOARD_DIR / "api_district_risk_map.csv"
district_map.to_csv(district_map_output, index=False)

print(f"Saved: {district_map_output}")

# --------------------------------------------------
# 4. Build crime type risk table
# --------------------------------------------------

print("\n[4/7] Building crime type risk table")

crime_type_risk_path = PROCESSED_DIR / "crime_type_risk_forecast_30day.csv"

if not crime_type_risk_path.exists():
    raise FileNotFoundError(
        "crime_type_risk_forecast_30day.csv not found. "
        "Run scripts\\08_train_xgboost.py first."
    )

crime_type_risk = pd.read_csv(crime_type_risk_path)

# Prevent tiny-volume categories from looking too dramatic.
# If predicted 30-day count is less than 100, mark confidence as Low Volume.
crime_type_risk["risk_confidence"] = crime_type_risk["predicted_30_day_crime_count"].apply(
    lambda x: "Low Volume" if x < 100 else "Stable"
)

crime_type_risk["risk_direction"] = crime_type_risk["risk_percent_vs_recent_baseline"].apply(
    lambda x: "Above Recent Baseline" if x > 0 else "Below Recent Baseline"
)

crime_type_risk["display_risk_percent"] = crime_type_risk[
    "risk_percent_vs_recent_baseline"
].round(2)

crime_type_risk = crime_type_risk[
    [
        "primary_type",
        "predicted_30_day_crime_count",
        "expected_30_day_count",
        "display_risk_percent",
        "risk_direction",
        "risk_level",
        "risk_confidence",
    ]
].copy()

crime_type_risk_output = DASHBOARD_DIR / "api_crime_type_risk.csv"
crime_type_risk.to_csv(crime_type_risk_output, index=False)

print(f"Saved: {crime_type_risk_output}")

# --------------------------------------------------
# 5. Build high-risk time period table
# --------------------------------------------------

print("\n[5/7] Building high-risk time period table")

tables = [row[0] for row in con.execute("SHOW TABLES;").fetchall()]

if "high_risk_time_periods_v2" in tables:
    high_risk_table = "high_risk_time_periods_v2"
else:
    high_risk_table = "high_risk_time_periods"

high_risk_periods = con.execute(f"""
SELECT
    district,
    primary_type,
    day_of_week,
    hour,
    crime_count,
    share_of_group_percent,
    risk_percent_vs_group_avg,
    CASE
        WHEN risk_percent_vs_group_avg >= 75 THEN 'Very High'
        WHEN risk_percent_vs_group_avg >= 40 THEN 'High'
        WHEN risk_percent_vs_group_avg >= 15 THEN 'Moderate'
        ELSE 'Normal'
    END AS risk_level
FROM {high_risk_table}
ORDER BY risk_percent_vs_group_avg DESC
LIMIT 500;
""").df()

day_name_map = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}

high_risk_periods["day_name"] = high_risk_periods["day_of_week"].map(day_name_map)

high_risk_periods["hour_label"] = high_risk_periods["hour"].apply(
    lambda h: f"{int(h):02d}:00"
)

high_risk_periods = high_risk_periods[
    [
        "district",
        "primary_type",
        "day_name",
        "hour_label",
        "crime_count",
        "share_of_group_percent",
        "risk_percent_vs_group_avg",
        "risk_level",
    ]
].copy()

high_risk_output = DASHBOARD_DIR / "api_high_risk_time_periods.csv"
high_risk_periods.to_csv(high_risk_output, index=False)

print(f"Saved: {high_risk_output}")

# --------------------------------------------------
# 6. Historical trend samples for dashboard charts
# --------------------------------------------------

print("\n[6/7] Building historical trend files")

citywide_history = con.execute("""
SELECT
    date,
    crime_count
FROM citywide_model_ready
ORDER BY date;
""").df()

citywide_history["date"] = pd.to_datetime(citywide_history["date"]).dt.strftime("%Y-%m-%d")

# Keep full history available but dashboard can choose recent range.
citywide_history_output = DASHBOARD_DIR / "api_citywide_history.csv"
citywide_history.to_csv(citywide_history_output, index=False)

district_history = con.execute("""
SELECT
    date,
    district,
    crime_count
FROM district_model_ready
ORDER BY date, district;
""").df()

district_history["date"] = pd.to_datetime(district_history["date"]).dt.strftime("%Y-%m-%d")

district_history_output = DASHBOARD_DIR / "api_district_history.csv"
district_history.to_csv(district_history_output, index=False)

crime_type_history = con.execute("""
SELECT
    date,
    primary_type,
    crime_count
FROM crime_type_model_ready
ORDER BY date, primary_type;
""").df()

crime_type_history["date"] = pd.to_datetime(crime_type_history["date"]).dt.strftime("%Y-%m-%d")

crime_type_history_output = DASHBOARD_DIR / "api_crime_type_history.csv"
crime_type_history.to_csv(crime_type_history_output, index=False)

print(f"Saved: {citywide_history_output}")
print(f"Saved: {district_history_output}")
print(f"Saved: {crime_type_history_output}")

# --------------------------------------------------
# 7. Build dashboard summary JSON
# --------------------------------------------------

print("\n[7/7] Building dashboard summary JSON")

date_range = con.execute("""
SELECT
    MIN(crime_date) AS min_date,
    MAX(crime_date) AS max_date,
    COUNT(*) AS total_records
FROM crimes_clean;
""").fetchone()

safe_range = con.execute("""
SELECT
    MIN(date) AS min_date,
    MAX(date) AS max_date,
    COUNT(*) AS total_days
FROM citywide_model_ready;
""").fetchone()

top_district = district_map.iloc[0].to_dict()
top_crime_type = crime_type_risk.iloc[0].to_dict()

forecast_7_total = int(citywide_7day["predicted_crime_count"].sum())
forecast_30_total = int(citywide_30day["predicted_crime_count"].sum())

summary = {
    "project_name": "Chicago Crime Forecasting & Risk Analysis Dashboard",
    "raw_data_start_date": str(date_range[0]),
    "raw_data_end_date": str(date_range[1]),
    "model_training_start_date": str(safe_range[0]),
    "model_training_end_date": str(safe_range[1]),
    "total_clean_records": int(date_range[2]),
    "total_training_days": int(safe_range[2]),
    "best_citywide_model": str(best_citywide_model),
    "best_citywide_model_group": str(best_citywide_group),
    "forecast_7_day_total": forecast_7_total,
    "forecast_30_day_total": forecast_30_total,
    "top_district_by_risk": {
        "district": int(top_district["district"]),
        "predicted_30_day_crime_count": int(top_district["predicted_30_day_crime_count"]),
        "risk_percent": float(top_district["display_risk_percent"]),
        "risk_level": str(top_district["risk_level"]),
    },
    "top_crime_type_by_risk": {
        "primary_type": str(top_crime_type["primary_type"]),
        "predicted_30_day_crime_count": int(top_crime_type["predicted_30_day_crime_count"]),
        "risk_percent": float(top_crime_type["display_risk_percent"]),
        "risk_level": str(top_crime_type["risk_level"]),
        "risk_confidence": str(top_crime_type["risk_confidence"]),
    },
}

summary_output = DASHBOARD_DIR / "api_dashboard_summary.json"

with open(summary_output, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=4)

print(f"Saved: {summary_output}")

# Also save model comparison for dashboard
model_comparison_output = DASHBOARD_DIR / "api_model_comparison.csv"
model_comparison.to_csv(model_comparison_output, index=False)
print(f"Saved: {model_comparison_output}")

con.close()

print("\n" + "=" * 80)
print("DASHBOARD OUTPUT BUILD COMPLETE")
print("=" * 80)
print(f"Dashboard files saved in: {DASHBOARD_DIR}")
print("=" * 80)
