from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import time

ROOT_DIR = Path(__file__).resolve().parents[1]

DB_PATH = ROOT_DIR / "db" / "chicago_crime.duckdb"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("BASELINE CITYWIDE FORECAST MODEL")
print("=" * 80)

if not DB_PATH.exists():
    raise FileNotFoundError(f"DuckDB database not found: {DB_PATH}")

start_time = time.time()

con = duckdb.connect(str(DB_PATH))

# --------------------------------------------------
# 1. Load model-ready citywide data
# --------------------------------------------------

print("\n[1/6] Loading citywide_model_ready table")

tables = [row[0] for row in con.execute("SHOW TABLES;").fetchall()]

if "citywide_model_ready" not in tables:
    raise RuntimeError(
        "citywide_model_ready table not found. "
        "Run scripts\\04_build_model_ready_tables.py first."
    )

df = con.execute("""
SELECT
    date,
    crime_count
FROM citywide_model_ready
ORDER BY date;
""").df()

df["date"] = pd.to_datetime(df["date"])
df["crime_count"] = df["crime_count"].astype(float)

print(f"Rows loaded: {len(df):,}")
print("Min date:", df["date"].min().date())
print("Max date:", df["date"].max().date())

# --------------------------------------------------
# 2. Create baseline prediction features
# --------------------------------------------------

print("\n[2/6] Creating baseline prediction columns")

df["pred_lag_1"] = df["crime_count"].shift(1)
df["pred_lag_7"] = df["crime_count"].shift(7)
df["pred_rolling_7"] = df["crime_count"].shift(1).rolling(window=7).mean()
df["pred_rolling_30"] = df["crime_count"].shift(1).rolling(window=30).mean()

df = df.dropna().reset_index(drop=True)

# --------------------------------------------------
# 3. Train/test split
# --------------------------------------------------
# For time series, we do not randomly split.
# We use the latest 90 days as the test period.

print("\n[3/6] Creating time-series test split")

TEST_DAYS = 90

train_df = df.iloc[:-TEST_DAYS].copy()
test_df = df.iloc[-TEST_DAYS:].copy()

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows: {len(test_df):,}")
print("Test start:", test_df["date"].min().date())
print("Test end:", test_df["date"].max().date())

# --------------------------------------------------
# 4. Evaluation functions
# --------------------------------------------------

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


# --------------------------------------------------
# 5. Compare baseline models
# --------------------------------------------------

print("\n[4/6] Evaluating baseline models")

baseline_models = {
    "Naive Previous Day": "pred_lag_1",
    "Seasonal Naive Previous Week": "pred_lag_7",
    "Rolling Mean 7 Days": "pred_rolling_7",
    "Rolling Mean 30 Days": "pred_rolling_30",
}

results = []

for model_name, pred_col in baseline_models.items():
    y_true = test_df["crime_count"].values
    y_pred = test_df[pred_col].values

    result = {
        "model": model_name,
        "mae": round(mae(y_true, y_pred), 2),
        "rmse": round(rmse(y_true, y_pred), 2),
        "mape": round(mape(y_true, y_pred), 2),
        "r2": round(r2_score_manual(y_true, y_pred), 4),
    }

    results.append(result)

comparison_df = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)

print("\nBaseline model comparison:")
print(comparison_df)

best_model_name = comparison_df.iloc[0]["model"]
best_model_col = baseline_models[best_model_name]

print("\nBest baseline model:", best_model_name)

# --------------------------------------------------
# 6. Save test predictions
# --------------------------------------------------

print("\n[5/6] Saving baseline test predictions")

test_predictions = test_df[
    [
        "date",
        "crime_count",
        "pred_lag_1",
        "pred_lag_7",
        "pred_rolling_7",
        "pred_rolling_30",
    ]
].copy()

test_predictions = test_predictions.rename(
    columns={
        "crime_count": "actual_crime_count",
        "pred_lag_1": "naive_previous_day_prediction",
        "pred_lag_7": "seasonal_previous_week_prediction",
        "pred_rolling_7": "rolling_mean_7_prediction",
        "pred_rolling_30": "rolling_mean_30_prediction",
    }
)

test_predictions["best_baseline_model"] = best_model_name
test_predictions["best_baseline_prediction"] = test_df[best_model_col].values

test_predictions_path = PROCESSED_DIR / "baseline_test_predictions.csv"
comparison_path = PROCESSED_DIR / "baseline_model_comparison.csv"

test_predictions.to_csv(test_predictions_path, index=False)
comparison_df.to_csv(comparison_path, index=False)

print(f"Saved: {test_predictions_path}")
print(f"Saved: {comparison_path}")

# --------------------------------------------------
# 7. Generate future 7-day and 30-day forecast
# --------------------------------------------------

print("\n[6/6] Generating future 7-day and 30-day baseline forecast")

history = df[["date", "crime_count"]].copy()
future_rows = []

last_date = history["date"].max()

# Iterative rolling 7-day baseline forecast.
# For each future day, use the average of the latest 7 known/predicted days.

for step in range(1, 31):
    future_date = last_date + pd.Timedelta(days=step)

    latest_7_values = history["crime_count"].tail(7)
    predicted_count = latest_7_values.mean()

    predicted_count = max(0, round(predicted_count))

    future_rows.append(
        {
            "date": future_date,
            "forecast_day": step,
            "predicted_crime_count": predicted_count,
            "model": "Baseline Rolling 7-Day Iterative Forecast",
        }
    )

    new_row = pd.DataFrame(
        {
            "date": [future_date],
            "crime_count": [predicted_count],
        }
    )

    history = pd.concat([history, new_row], ignore_index=True)

forecast_30_df = pd.DataFrame(future_rows)
forecast_7_df = forecast_30_df.head(7).copy()

forecast_7_path = PROCESSED_DIR / "citywide_7day_forecast_baseline.csv"
forecast_30_path = PROCESSED_DIR / "citywide_30day_forecast_baseline.csv"

forecast_7_df.to_csv(forecast_7_path, index=False)
forecast_30_df.to_csv(forecast_30_path, index=False)

print(f"Saved: {forecast_7_path}")
print(f"Saved: {forecast_30_path}")

print("\nNext 7-day baseline forecast:")
print(forecast_7_df)

print("\nNext 30-day baseline forecast:")
print(forecast_30_df)

# --------------------------------------------------
# 8. Save outputs into DuckDB tables too
# --------------------------------------------------

con.execute("DROP TABLE IF EXISTS baseline_model_comparison;")
con.execute("DROP TABLE IF EXISTS baseline_test_predictions;")
con.execute("DROP TABLE IF EXISTS citywide_7day_forecast_baseline;")
con.execute("DROP TABLE IF EXISTS citywide_30day_forecast_baseline;")

con.register("comparison_df_view", comparison_df)
con.register("test_predictions_view", test_predictions)
con.register("forecast_7_view", forecast_7_df)
con.register("forecast_30_view", forecast_30_df)

con.execute("CREATE TABLE baseline_model_comparison AS SELECT * FROM comparison_df_view;")
con.execute("CREATE TABLE baseline_test_predictions AS SELECT * FROM test_predictions_view;")
con.execute("CREATE TABLE citywide_7day_forecast_baseline AS SELECT * FROM forecast_7_view;")
con.execute("CREATE TABLE citywide_30day_forecast_baseline AS SELECT * FROM forecast_30_view;")

con.close()

elapsed = (time.time() - start_time) / 60

print("\n" + "=" * 80)
print("BASELINE FORECAST COMPLETE")
print("=" * 80)
print(f"Runtime: {elapsed:.2f} minutes")
print(f"Best baseline model: {best_model_name}")
print(f"Outputs saved in: {PROCESSED_DIR}")
print("=" * 80)
