from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import joblib
import time
import os
import json

from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]

DB_PATH = ROOT_DIR / "db" / "chicago_crime.duckdb"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("XGBOOST DISTRICT + CRIME TYPE FORECAST MODEL")
print("=" * 80)

if not DB_PATH.exists():
    raise FileNotFoundError(f"DuckDB database not found: {DB_PATH}")

start_time = time.time()


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mask = y_true != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def r2_score_manual(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return np.nan

    return 1 - (ss_res / ss_tot)


def assign_risk_level(risk_percent):
    if risk_percent >= 25:
        return "Very High"
    elif risk_percent >= 10:
        return "High"
    elif risk_percent >= 0:
        return "Moderate"
    else:
        return "Low"


# --------------------------------------------------
# 1. Load XGBoost feature table
# --------------------------------------------------

print("\n[1/9] Loading xgboost_features_daily table")

con = duckdb.connect(str(DB_PATH))

tables = [row[0] for row in con.execute("SHOW TABLES;").fetchall()]

if "xgboost_features_daily" not in tables:
    raise RuntimeError(
        "xgboost_features_daily table not found. "
        "Run scripts\\04_build_model_ready_tables.py first."
    )

df = con.execute("""
SELECT
    date,
    district,
    primary_type,
    crime_count,

    day,
    month,
    year,
    day_of_week,
    week_of_year,
    CAST(is_weekend AS INTEGER) AS is_weekend,

    lag_1,
    lag_7,
    lag_14,
    lag_30,
    rolling_mean_7,
    rolling_mean_30,
    rolling_std_7

FROM xgboost_features_daily
ORDER BY date, district, primary_type;
""").df()

df["date"] = pd.to_datetime(df["date"])
df["crime_count"] = df["crime_count"].astype(float)

print(f"Rows loaded: {len(df):,}")
print("Min date:", df["date"].min().date())
print("Max date:", df["date"].max().date())
print("Districts:", df["district"].nunique())
print("Crime types:", df["primary_type"].nunique())

# --------------------------------------------------
# 2. Encode crime types
# --------------------------------------------------

print("\n[2/9] Encoding categorical values")

crime_types = sorted(df["primary_type"].unique())
crime_type_map = {crime_type: idx for idx, crime_type in enumerate(crime_types)}
inverse_crime_type_map = {idx: crime_type for crime_type, idx in crime_type_map.items()}

df["primary_type_code"] = df["primary_type"].map(crime_type_map).astype(int)
df["district"] = df["district"].astype(int)

feature_columns = [
    "district",
    "primary_type_code",
    "day",
    "month",
    "year",
    "day_of_week",
    "week_of_year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_30",
    "rolling_std_7",
]

df[feature_columns] = df[feature_columns].fillna(0)

print("Feature columns:")
for col in feature_columns:
    print("-", col)

# --------------------------------------------------
# 3. Time-based train/test split
# --------------------------------------------------

print("\n[3/9] Creating time-series test split")

max_date = df["date"].max()
test_start_date = max_date - pd.Timedelta(days=89)

train_df = df[df["date"] < test_start_date].copy()
test_df = df[df["date"] >= test_start_date].copy()

print("Train max date:", train_df["date"].max().date())
print("Test start date:", test_df["date"].min().date())
print("Test end date:", test_df["date"].max().date())
print(f"Training rows: {len(train_df):,}")
print(f"Testing rows: {len(test_df):,}")

X_train = train_df[feature_columns]
y_train = train_df["crime_count"]

X_test = test_df[feature_columns]
y_test = test_df["crime_count"]

# --------------------------------------------------
# 4. Train XGBoost model
# --------------------------------------------------

print("\n[4/9] Training XGBoost model")
print("This may take a few minutes...")

model = XGBRegressor(
    objective="reg:squarederror",
    n_estimators=250,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.85,
    colsample_bytree=0.85,
    tree_method="hist",
    random_state=42,
    n_jobs=max(1, (os.cpu_count() or 4) - 1),
)

model.fit(X_train, y_train)

# --------------------------------------------------
# 5. Evaluate model
# --------------------------------------------------

print("\n[5/9] Evaluating XGBoost model")

test_pred = model.predict(X_test)
test_pred = np.maximum(0, test_pred)

row_level_metrics = {
    "model": "XGBoost District + Crime Type",
    "prediction_level": "district_crime_type_daily",
    "mae": round(mae(y_test.values, test_pred), 4),
    "rmse": round(rmse(y_test.values, test_pred), 4),
    "mape": round(mape(y_test.values, test_pred), 4),
    "r2": round(r2_score_manual(y_test.values, test_pred), 4),
}

print("\nRow-level XGBoost metrics:")
print(row_level_metrics)

test_predictions = test_df[
    [
        "date",
        "district",
        "primary_type",
        "crime_count",
    ]
].copy()

test_predictions = test_predictions.rename(
    columns={"crime_count": "actual_crime_count"}
)

test_predictions["predicted_crime_count"] = np.round(test_pred, 2)

# Aggregate test predictions by district
district_test_eval = (
    test_predictions
    .groupby(["date", "district"], as_index=False)
    .agg(
        actual_crime_count=("actual_crime_count", "sum"),
        predicted_crime_count=("predicted_crime_count", "sum"),
    )
)

district_level_metrics = {
    "model": "XGBoost District Forecast",
    "prediction_level": "district_daily",
    "mae": round(mae(district_test_eval["actual_crime_count"], district_test_eval["predicted_crime_count"]), 4),
    "rmse": round(rmse(district_test_eval["actual_crime_count"], district_test_eval["predicted_crime_count"]), 4),
    "mape": round(mape(district_test_eval["actual_crime_count"], district_test_eval["predicted_crime_count"]), 4),
    "r2": round(r2_score_manual(district_test_eval["actual_crime_count"], district_test_eval["predicted_crime_count"]), 4),
}

print("\nDistrict-level aggregated metrics:")
print(district_level_metrics)

crime_type_test_eval = (
    test_predictions
    .groupby(["date", "primary_type"], as_index=False)
    .agg(
        actual_crime_count=("actual_crime_count", "sum"),
        predicted_crime_count=("predicted_crime_count", "sum"),
    )
)

crime_type_level_metrics = {
    "model": "XGBoost Crime Type Forecast",
    "prediction_level": "crime_type_daily",
    "mae": round(mae(crime_type_test_eval["actual_crime_count"], crime_type_test_eval["predicted_crime_count"]), 4),
    "rmse": round(rmse(crime_type_test_eval["actual_crime_count"], crime_type_test_eval["predicted_crime_count"]), 4),
    "mape": round(mape(crime_type_test_eval["actual_crime_count"], crime_type_test_eval["predicted_crime_count"]), 4),
    "r2": round(r2_score_manual(crime_type_test_eval["actual_crime_count"], crime_type_test_eval["predicted_crime_count"]), 4),
}

print("\nCrime-type-level aggregated metrics:")
print(crime_type_level_metrics)

metrics_df = pd.DataFrame(
    [
        row_level_metrics,
        district_level_metrics,
        crime_type_level_metrics,
    ]
)

# --------------------------------------------------
# 6. Save model and metadata
# --------------------------------------------------

print("\n[6/9] Saving XGBoost model and metadata")

model_path = MODELS_DIR / "xgboost_model.joblib"
metadata_path = MODELS_DIR / "xgboost_metadata.joblib"
metadata_json_path = MODELS_DIR / "xgboost_metadata.json"

metadata = {
    "feature_columns": feature_columns,
    "crime_type_map": crime_type_map,
    "inverse_crime_type_map": inverse_crime_type_map,
    "max_training_date": str(max_date.date()),
}

joblib.dump(model, model_path)
joblib.dump(metadata, metadata_path)

with open(metadata_json_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=4)

print(f"Saved model: {model_path}")
print(f"Saved metadata: {metadata_path}")
print(f"Saved metadata JSON: {metadata_json_path}")

# --------------------------------------------------
# 7. Generate future 30-day district + crime type forecast
# --------------------------------------------------

print("\n[7/9] Generating future 30-day district + crime type forecast")

history_df = df[["date", "district", "primary_type", "crime_count"]].copy()

combos = (
    df[["district", "primary_type", "primary_type_code"]]
    .drop_duplicates()
    .sort_values(["district", "primary_type"])
    .reset_index(drop=True)
)

history_by_key = {}

for (district, primary_type), group in history_df.groupby(["district", "primary_type"]):
    key = (int(district), primary_type)
    history_by_key[key] = group.sort_values("date")["crime_count"].astype(float).tolist()

future_rows = []

for step in range(1, 31):
    future_date = max_date + pd.Timedelta(days=step)

    # Match DuckDB dow convention:
    # Sunday = 0, Monday = 1, ..., Saturday = 6
    day_of_week = (future_date.weekday() + 1) % 7

    feature_rows = []

    for row in combos.itertuples(index=False):
        district = int(row.district)
        primary_type = row.primary_type
        primary_type_code = int(row.primary_type_code)

        key = (district, primary_type)
        values = history_by_key[key]

        latest_7 = values[-7:]
        latest_30 = values[-30:]

        feature_rows.append(
            {
                "date": future_date,
                "forecast_day": step,
                "district": district,
                "primary_type": primary_type,
                "primary_type_code": primary_type_code,

                "day": future_date.day,
                "month": future_date.month,
                "year": future_date.year,
                "day_of_week": day_of_week,
                "week_of_year": int(future_date.isocalendar().week),
                "is_weekend": 1 if day_of_week in [0, 6] else 0,

                "lag_1": values[-1],
                "lag_7": values[-7],
                "lag_14": values[-14],
                "lag_30": values[-30],
                "rolling_mean_7": float(np.mean(latest_7)),
                "rolling_mean_30": float(np.mean(latest_30)),
                "rolling_std_7": float(np.std(latest_7, ddof=1)) if len(latest_7) > 1 else 0.0,
            }
        )

    future_feature_df = pd.DataFrame(feature_rows)
    future_feature_df[feature_columns] = future_feature_df[feature_columns].fillna(0)

    predicted = model.predict(future_feature_df[feature_columns])
    predicted = np.maximum(0, predicted)
    predicted_rounded = np.round(predicted).astype(int)

    future_feature_df["predicted_crime_count"] = predicted_rounded

    for row in future_feature_df.itertuples(index=False):
        key = (int(row.district), row.primary_type)
        history_by_key[key].append(float(row.predicted_crime_count))

    future_rows.append(
        future_feature_df[
            [
                "date",
                "forecast_day",
                "district",
                "primary_type",
                "predicted_crime_count",
            ]
        ].copy()
    )

forecast_detail_30_df = pd.concat(future_rows, ignore_index=True)

print(f"Future forecast rows: {len(forecast_detail_30_df):,}")

# --------------------------------------------------
# 8. Aggregate forecasts and build risk percentages
# --------------------------------------------------

print("\n[8/9] Aggregating forecasts and calculating risk percentages")

district_forecast_30_df = (
    forecast_detail_30_df
    .groupby(["date", "forecast_day", "district"], as_index=False)
    .agg(predicted_crime_count=("predicted_crime_count", "sum"))
)

crime_type_forecast_30_df = (
    forecast_detail_30_df
    .groupby(["date", "forecast_day", "primary_type"], as_index=False)
    .agg(predicted_crime_count=("predicted_crime_count", "sum"))
)

citywide_forecast_30_df = (
    forecast_detail_30_df
    .groupby(["date", "forecast_day"], as_index=False)
    .agg(predicted_crime_count=("predicted_crime_count", "sum"))
)

citywide_forecast_30_df["model"] = "XGBoost District + Crime Type Aggregated"
citywide_forecast_7_df = citywide_forecast_30_df.head(7).copy()

district_30_total = (
    district_forecast_30_df
    .groupby("district", as_index=False)
    .agg(predicted_30_day_crime_count=("predicted_crime_count", "sum"))
)

crime_type_30_total = (
    crime_type_forecast_30_df
    .groupby("primary_type", as_index=False)
    .agg(predicted_30_day_crime_count=("predicted_crime_count", "sum"))
)

latest_date_sql = max_date.strftime("%Y-%m-%d")

district_baseline_df = con.execute(f"""
SELECT
    district,
    ROUND(AVG(crime_count) * 30, 2) AS expected_30_day_count
FROM district_model_ready
WHERE date > DATE '{latest_date_sql}' - INTERVAL 365 DAY
  AND date <= DATE '{latest_date_sql}'
GROUP BY district
ORDER BY district;
""").df()

crime_type_baseline_df = con.execute(f"""
SELECT
    primary_type,
    ROUND(AVG(crime_count) * 30, 2) AS expected_30_day_count
FROM crime_type_model_ready
WHERE date > DATE '{latest_date_sql}' - INTERVAL 365 DAY
  AND date <= DATE '{latest_date_sql}'
GROUP BY primary_type
ORDER BY primary_type;
""").df()

district_risk_forecast_df = district_30_total.merge(
    district_baseline_df,
    on="district",
    how="left",
)

district_risk_forecast_df["risk_percent_vs_recent_baseline"] = (
    100.0
    * (
        district_risk_forecast_df["predicted_30_day_crime_count"]
        - district_risk_forecast_df["expected_30_day_count"]
    )
    / district_risk_forecast_df["expected_30_day_count"]
).round(2)

district_risk_forecast_df["risk_level"] = district_risk_forecast_df[
    "risk_percent_vs_recent_baseline"
].apply(assign_risk_level)

district_risk_forecast_df = district_risk_forecast_df.sort_values(
    "risk_percent_vs_recent_baseline",
    ascending=False,
).reset_index(drop=True)

crime_type_risk_forecast_df = crime_type_30_total.merge(
    crime_type_baseline_df,
    on="primary_type",
    how="left",
)

crime_type_risk_forecast_df["risk_percent_vs_recent_baseline"] = (
    100.0
    * (
        crime_type_risk_forecast_df["predicted_30_day_crime_count"]
        - crime_type_risk_forecast_df["expected_30_day_count"]
    )
    / crime_type_risk_forecast_df["expected_30_day_count"]
).round(2)

crime_type_risk_forecast_df["risk_level"] = crime_type_risk_forecast_df[
    "risk_percent_vs_recent_baseline"
].apply(assign_risk_level)

crime_type_risk_forecast_df = crime_type_risk_forecast_df.sort_values(
    "risk_percent_vs_recent_baseline",
    ascending=False,
).reset_index(drop=True)

# District map points using historical median latitude/longitude
district_centroids_df = con.execute("""
SELECT
    district,
    MEDIAN(latitude) AS latitude,
    MEDIAN(longitude) AS longitude
FROM crimes_clean
WHERE district IS NOT NULL
  AND latitude IS NOT NULL
  AND longitude IS NOT NULL
  AND latitude BETWEEN 41.5 AND 42.1
  AND longitude BETWEEN -88.0 AND -87.4
GROUP BY district
ORDER BY district;
""").df()

district_map_points_df = district_risk_forecast_df.merge(
    district_centroids_df,
    on="district",
    how="left",
)

print("\nDistrict forecast risk preview:")
print(district_risk_forecast_df.head(15))

print("\nCrime type forecast risk preview:")
print(crime_type_risk_forecast_df.head(15))

# --------------------------------------------------
# 9. Save files and DuckDB tables
# --------------------------------------------------

print("\n[9/9] Saving XGBoost outputs")

paths = {
    "xgboost_model_comparison": PROCESSED_DIR / "xgboost_model_comparison.csv",
    "xgboost_test_predictions": PROCESSED_DIR / "xgboost_test_predictions.csv",
    "xgboost_district_test_eval": PROCESSED_DIR / "xgboost_district_test_eval.csv",
    "xgboost_crime_type_test_eval": PROCESSED_DIR / "xgboost_crime_type_test_eval.csv",

    "xgboost_forecast_detail_30day": PROCESSED_DIR / "xgboost_forecast_detail_30day.csv",
    "district_30day_forecast_xgboost": PROCESSED_DIR / "district_30day_forecast_xgboost.csv",
    "crime_type_30day_forecast_xgboost": PROCESSED_DIR / "crime_type_30day_forecast_xgboost.csv",
    "citywide_7day_forecast_xgboost": PROCESSED_DIR / "citywide_7day_forecast_xgboost.csv",
    "citywide_30day_forecast_xgboost": PROCESSED_DIR / "citywide_30day_forecast_xgboost.csv",

    "district_risk_forecast_30day": PROCESSED_DIR / "district_risk_forecast_30day.csv",
    "crime_type_risk_forecast_30day": PROCESSED_DIR / "crime_type_risk_forecast_30day.csv",
    "district_map_points": PROCESSED_DIR / "district_map_points.csv",
}

metrics_df.to_csv(paths["xgboost_model_comparison"], index=False)
test_predictions.to_csv(paths["xgboost_test_predictions"], index=False)
district_test_eval.to_csv(paths["xgboost_district_test_eval"], index=False)
crime_type_test_eval.to_csv(paths["xgboost_crime_type_test_eval"], index=False)

forecast_detail_30_df.to_csv(paths["xgboost_forecast_detail_30day"], index=False)
district_forecast_30_df.to_csv(paths["district_30day_forecast_xgboost"], index=False)
crime_type_forecast_30_df.to_csv(paths["crime_type_30day_forecast_xgboost"], index=False)
citywide_forecast_7_df.to_csv(paths["citywide_7day_forecast_xgboost"], index=False)
citywide_forecast_30_df.to_csv(paths["citywide_30day_forecast_xgboost"], index=False)

district_risk_forecast_df.to_csv(paths["district_risk_forecast_30day"], index=False)
crime_type_risk_forecast_df.to_csv(paths["crime_type_risk_forecast_30day"], index=False)
district_map_points_df.to_csv(paths["district_map_points"], index=False)

for name, path in paths.items():
    print(f"Saved {name}: {path}")

# Save into DuckDB
tables_to_save = {
    "xgboost_model_comparison": metrics_df,
    "xgboost_test_predictions": test_predictions,
    "xgboost_district_test_eval": district_test_eval,
    "xgboost_crime_type_test_eval": crime_type_test_eval,

    "xgboost_forecast_detail_30day": forecast_detail_30_df,
    "district_30day_forecast_xgboost": district_forecast_30_df,
    "crime_type_30day_forecast_xgboost": crime_type_forecast_30_df,
    "citywide_7day_forecast_xgboost": citywide_forecast_7_df,
    "citywide_30day_forecast_xgboost": citywide_forecast_30_df,

    "district_risk_forecast_30day": district_risk_forecast_df,
    "crime_type_risk_forecast_30day": crime_type_risk_forecast_df,
    "district_map_points": district_map_points_df,
}

for table_name, table_df in tables_to_save.items():
    con.execute(f"DROP TABLE IF EXISTS {table_name};")
    con.register(f"{table_name}_view", table_df)
    con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {table_name}_view;")

con.close()

elapsed = (time.time() - start_time) / 60

print("\n" + "=" * 80)
print("XGBOOST FORECAST COMPLETE")
print("=" * 80)
print(f"Runtime: {elapsed:.2f} minutes")
print(f"Model saved at: {model_path}")
print(f"Metadata saved at: {metadata_path}")
print(f"Outputs saved in: {PROCESSED_DIR}")
print("=" * 80)