from pathlib import Path
import duckdb
import pandas as pd
import numpy as np
import time
import warnings
import joblib

from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parents[1]

DB_PATH = ROOT_DIR / "db" / "chicago_crime.duckdb"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("SARIMA CITYWIDE FORECAST MODEL")
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


# --------------------------------------------------
# 1. Load citywide data
# --------------------------------------------------

print("\n[1/7] Loading citywide_model_ready table")

con = duckdb.connect(str(DB_PATH))

df = con.execute("""
SELECT
    date,
    crime_count
FROM citywide_model_ready
ORDER BY date;
""").df()

con.close()

df["date"] = pd.to_datetime(df["date"])
df["crime_count"] = df["crime_count"].astype(float)

print(f"Rows loaded: {len(df):,}")
print("Min date:", df["date"].min().date())
print("Max date:", df["date"].max().date())

# --------------------------------------------------
# 2. Use full historical data
# --------------------------------------------------

print("\n[2/7] Using full historical data for SARIMA")
print("Training history starts:", df["date"].min().date())
print("Training history ends:", df["date"].max().date())
print(f"Rows used for modeling: {len(df):,}")

# --------------------------------------------------
# 3. Train/test split
# --------------------------------------------------

print("\n[3/7] Creating time-series test split")

TEST_DAYS = 90

train_df = df.iloc[:-TEST_DAYS].copy()
test_df = df.iloc[-TEST_DAYS:].copy()

train_series = train_df.set_index("date")["crime_count"].asfreq("D")
test_series = test_df.set_index("date")["crime_count"].asfreq("D")

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows: {len(test_df):,}")
print("Test start:", test_df["date"].min().date())
print("Test end:", test_df["date"].max().date())

# --------------------------------------------------
# 4. Train SARIMA candidates
# --------------------------------------------------
# Seasonal period = 7 because crime has weekly patterns.

print("\n[4/7] Training SARIMA candidate models")

candidate_configs = [
    ((1, 1, 1), (1, 0, 1, 7)),
    ((2, 1, 1), (1, 0, 1, 7)),
    ((1, 1, 2), (1, 0, 1, 7)),
    ((2, 1, 2), (1, 0, 1, 7)),
    ((1, 1, 1), (1, 1, 1, 7)),
    ((2, 1, 1), (1, 1, 1, 7)),
]

candidate_results = []

best_model_fit = None
best_order = None
best_seasonal_order = None
best_mae = float("inf")
best_forecast = None

for order, seasonal_order in candidate_configs:
    print(f"\nTrying SARIMA order={order}, seasonal_order={seasonal_order}")

    try:
        model = SARIMAX(
            train_series,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        model_fit = model.fit(disp=False, maxiter=100)

        forecast_result = model_fit.get_forecast(steps=TEST_DAYS)
        forecast_values = forecast_result.predicted_mean.values
        forecast_values = np.maximum(0, forecast_values)

        current_mae = mae(test_series.values, forecast_values)
        current_rmse = rmse(test_series.values, forecast_values)
        current_mape = mape(test_series.values, forecast_values)
        current_r2 = r2_score_manual(test_series.values, forecast_values)

        result = {
            "model": f"SARIMA{order}x{seasonal_order}",
            "order": str(order),
            "seasonal_order": str(seasonal_order),
            "mae": round(current_mae, 2),
            "rmse": round(current_rmse, 2),
            "mape": round(current_mape, 2),
            "r2": round(current_r2, 4),
            "aic": round(model_fit.aic, 2),
        }

        candidate_results.append(result)

        print(
            f"MAE: {current_mae:.2f} | "
            f"RMSE: {current_rmse:.2f} | "
            f"MAPE: {current_mape:.2f}% | "
            f"R2: {current_r2:.4f} | "
            f"AIC: {model_fit.aic:.2f}"
        )

        if current_mae < best_mae:
            best_mae = current_mae
            best_order = order
            best_seasonal_order = seasonal_order
            best_model_fit = model_fit
            best_forecast = forecast_values

    except Exception as e:
        print(f"Failed for SARIMA{order}x{seasonal_order}: {e}")

if best_model_fit is None:
    raise RuntimeError("All SARIMA candidate models failed.")

comparison_df = pd.DataFrame(candidate_results).sort_values("mae").reset_index(drop=True)

print("\nSARIMA model comparison:")
print(comparison_df)

print("\nBest SARIMA order:", best_order)
print("Best seasonal order:", best_seasonal_order)

# --------------------------------------------------
# 5. Save test predictions
# --------------------------------------------------

print("\n[5/7] Saving SARIMA test predictions")

test_predictions = test_df[["date", "crime_count"]].copy()
test_predictions = test_predictions.rename(columns={"crime_count": "actual_crime_count"})
test_predictions["predicted_crime_count"] = np.round(best_forecast).astype(int)
test_predictions["model"] = f"SARIMA{best_order}x{best_seasonal_order}"

sarima_comparison_path = PROCESSED_DIR / "sarima_model_comparison.csv"
sarima_test_predictions_path = PROCESSED_DIR / "sarima_test_predictions.csv"

comparison_df.to_csv(sarima_comparison_path, index=False)
test_predictions.to_csv(sarima_test_predictions_path, index=False)

print(f"Saved: {sarima_comparison_path}")
print(f"Saved: {sarima_test_predictions_path}")

# --------------------------------------------------
# 6. Train final SARIMA model on full history
# --------------------------------------------------

print("\n[6/7] Training final SARIMA model on full historical data")

full_series = df.set_index("date")["crime_count"].asfreq("D")

final_model = SARIMAX(
    full_series,
    order=best_order,
    seasonal_order=best_seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)

final_model_fit = final_model.fit(disp=False, maxiter=100)

model_path = MODELS_DIR / "sarima_model.pkl"
joblib.dump(final_model_fit, model_path)

print(f"Saved model: {model_path}")

# --------------------------------------------------
# 7. Generate future forecast
# --------------------------------------------------

print("\n[7/7] Generating future 7-day and 30-day SARIMA forecast")

future_steps = 30
forecast_result = final_model_fit.get_forecast(steps=future_steps)

future_forecast = forecast_result.predicted_mean.values
future_forecast = np.maximum(0, future_forecast)

conf_int = forecast_result.conf_int()
lower_bound = np.maximum(0, conf_int.iloc[:, 0].values)
upper_bound = np.maximum(0, conf_int.iloc[:, 1].values)

last_date = df["date"].max()

future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=future_steps,
    freq="D",
)

forecast_30_df = pd.DataFrame({
    "date": future_dates,
    "forecast_day": range(1, future_steps + 1),
    "predicted_crime_count": np.round(future_forecast).astype(int),
    "lower_bound": np.round(lower_bound).astype(int),
    "upper_bound": np.round(upper_bound).astype(int),
    "model": f"SARIMA{best_order}x{best_seasonal_order}",
})

forecast_7_df = forecast_30_df.head(7).copy()

forecast_7_path = PROCESSED_DIR / "citywide_7day_forecast_sarima.csv"
forecast_30_path = PROCESSED_DIR / "citywide_30day_forecast_sarima.csv"

forecast_7_df.to_csv(forecast_7_path, index=False)
forecast_30_df.to_csv(forecast_30_path, index=False)

print(f"Saved: {forecast_7_path}")
print(f"Saved: {forecast_30_path}")

print("\nNext 7-day SARIMA forecast:")
print(forecast_7_df)

print("\nNext 30-day SARIMA forecast:")
print(forecast_30_df)

# --------------------------------------------------
# Save SARIMA outputs into DuckDB
# --------------------------------------------------

con = duckdb.connect(str(DB_PATH))

con.execute("DROP TABLE IF EXISTS sarima_model_comparison;")
con.execute("DROP TABLE IF EXISTS sarima_test_predictions;")
con.execute("DROP TABLE IF EXISTS citywide_7day_forecast_sarima;")
con.execute("DROP TABLE IF EXISTS citywide_30day_forecast_sarima;")

con.register("sarima_comparison_view", comparison_df)
con.register("sarima_test_predictions_view", test_predictions)
con.register("sarima_forecast_7_view", forecast_7_df)
con.register("sarima_forecast_30_view", forecast_30_df)

con.execute("CREATE TABLE sarima_model_comparison AS SELECT * FROM sarima_comparison_view;")
con.execute("CREATE TABLE sarima_test_predictions AS SELECT * FROM sarima_test_predictions_view;")
con.execute("CREATE TABLE citywide_7day_forecast_sarima AS SELECT * FROM sarima_forecast_7_view;")
con.execute("CREATE TABLE citywide_30day_forecast_sarima AS SELECT * FROM sarima_forecast_30_view;")

# --------------------------------------------------
# Update combined model comparison
# --------------------------------------------------

combined_parts = []

baseline_path = PROCESSED_DIR / "baseline_model_comparison.csv"
arima_path = PROCESSED_DIR / "arima_model_comparison.csv"

if baseline_path.exists():
    baseline_df = pd.read_csv(baseline_path).sort_values("mae").head(1).copy()
    baseline_df["model_group"] = "Baseline"
    combined_parts.append(baseline_df)

if arima_path.exists():
    arima_df = pd.read_csv(arima_path).sort_values("mae").head(1).copy()
    arima_df["model_group"] = "ARIMA"
    combined_parts.append(arima_df)

best_sarima = comparison_df.sort_values("mae").head(1).copy()
best_sarima["model_group"] = "SARIMA"
combined_parts.append(best_sarima)

combined_df = pd.concat(combined_parts, ignore_index=True, sort=False)
combined_df = combined_df.sort_values("mae").reset_index(drop=True)

combined_path = PROCESSED_DIR / "model_comparison_current.csv"
combined_df.to_csv(combined_path, index=False)

con.execute("DROP TABLE IF EXISTS model_comparison_current;")
con.register("combined_model_comparison_view", combined_df)
con.execute("CREATE TABLE model_comparison_current AS SELECT * FROM combined_model_comparison_view;")

con.close()

print(f"\nSaved combined comparison: {combined_path}")
print("\nCurrent best model comparison:")
print(combined_df)

elapsed = (time.time() - start_time) / 60

print("\n" + "=" * 80)
print("SARIMA FORECAST COMPLETE")
print("=" * 80)
print(f"Runtime: {elapsed:.2f} minutes")
print(f"Best SARIMA order: {best_order}")
print(f"Best seasonal order: {best_seasonal_order}")
print(f"Model saved at: {model_path}")
print(f"Outputs saved in: {PROCESSED_DIR}")
print("=" * 80)
