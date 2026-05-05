from pathlib import Path
import duckdb
import os
import time

ROOT_DIR = Path(__file__).resolve().parents[1]

DB_PATH = ROOT_DIR / "db" / "chicago_crime.duckdb"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("BUILDING MODEL-READY TABLES")
print("=" * 80)

if not DB_PATH.exists():
    raise FileNotFoundError(f"DuckDB database not found: {DB_PATH}")

start_time = time.time()

con = duckdb.connect(str(DB_PATH))

threads = max(1, os.cpu_count() or 4)
con.execute(f"PRAGMA threads={threads};")
con.execute("PRAGMA enable_progress_bar;")

print(f"Database: {DB_PATH}")
print(f"Using {threads} CPU threads")

# --------------------------------------------------
# 1. Find a safe training end date
# --------------------------------------------------
# The last date has only 1 record, so we automatically remove incomplete final days.
# Rule: use the latest date where crime_count is at least 50% of the previous 30-day median.

print("\n[1/7] Detecting safe training end date")

training_end_date = con.execute("""
WITH daily AS (
    SELECT
        date,
        crime_count,
        MEDIAN(crime_count) OVER (
            ORDER BY date
            ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
        ) AS previous_30_day_median
    FROM citywide_daily_counts
),
valid_days AS (
    SELECT
        date,
        crime_count,
        previous_30_day_median
    FROM daily
    WHERE previous_30_day_median IS NOT NULL
      AND crime_count >= previous_30_day_median * 0.50
)
SELECT MAX(date)
FROM valid_days;
""").fetchone()[0]

print("Safe training end date:", training_end_date)

# --------------------------------------------------
# 2. Create citywide model-ready table
# --------------------------------------------------

print("\n[2/7] Creating citywide_model_ready")

con.execute("DROP TABLE IF EXISTS citywide_model_ready;")

con.execute(f"""
CREATE TABLE citywide_model_ready AS
WITH calendar AS (
    SELECT *
    FROM generate_series(
        DATE '2001-01-01',
        DATE '{training_end_date}',
        INTERVAL 1 DAY
    ) AS t(date)
),
daily AS (
    SELECT
        date,
        crime_count
    FROM citywide_daily_counts
    WHERE date <= DATE '{training_end_date}'
)
SELECT
    c.date,
    COALESCE(d.crime_count, 0) AS crime_count,

    EXTRACT(day FROM c.date) AS day,
    EXTRACT(month FROM c.date) AS month,
    EXTRACT(year FROM c.date) AS year,
    EXTRACT(dow FROM c.date) AS day_of_week,
    EXTRACT(week FROM c.date) AS week_of_year,

    CASE
        WHEN EXTRACT(dow FROM c.date) IN (0, 6) THEN TRUE
        ELSE FALSE
    END AS is_weekend

FROM calendar c
LEFT JOIN daily d
    ON c.date = d.date
ORDER BY c.date;
""")

count = con.execute("SELECT COUNT(*) FROM citywide_model_ready;").fetchone()[0]
print(f"citywide_model_ready rows: {count:,}")

# --------------------------------------------------
# 3. Create district model-ready table
# --------------------------------------------------
# Active Chicago police districts used for dashboard/map.
# We exclude rare/invalid districts like 21 and 31.

print("\n[3/7] Creating district_model_ready")

con.execute("DROP TABLE IF EXISTS district_model_ready;")

con.execute(f"""
CREATE TABLE district_model_ready AS
WITH valid_districts AS (
    SELECT district
    FROM (
        VALUES
        (1), (2), (3), (4), (5), (6), (7), (8), (9), (10),
        (11), (12), (14), (15), (16), (17), (18), (19), (20),
        (22), (24), (25)
    ) AS t(district)
),
calendar AS (
    SELECT *
    FROM generate_series(
        DATE '2001-01-01',
        DATE '{training_end_date}',
        INTERVAL 1 DAY
    ) AS t(date)
),
grid AS (
    SELECT
        c.date,
        d.district
    FROM calendar c
    CROSS JOIN valid_districts d
),
daily AS (
    SELECT
        date,
        district,
        crime_count
    FROM district_daily_counts
    WHERE date <= DATE '{training_end_date}'
)
SELECT
    g.date,
    g.district,
    COALESCE(d.crime_count, 0) AS crime_count,

    EXTRACT(day FROM g.date) AS day,
    EXTRACT(month FROM g.date) AS month,
    EXTRACT(year FROM g.date) AS year,
    EXTRACT(dow FROM g.date) AS day_of_week,
    EXTRACT(week FROM g.date) AS week_of_year,

    CASE
        WHEN EXTRACT(dow FROM g.date) IN (0, 6) THEN TRUE
        ELSE FALSE
    END AS is_weekend

FROM grid g
LEFT JOIN daily d
    ON g.date = d.date
   AND g.district = d.district
ORDER BY g.date, g.district;
""")

count = con.execute("SELECT COUNT(*) FROM district_model_ready;").fetchone()[0]
print(f"district_model_ready rows: {count:,}")

# --------------------------------------------------
# 4. Create crime type model-ready table
# --------------------------------------------------
# Use top 15 crime types for cleaner dashboard and forecasting.

print("\n[4/7] Creating crime_type_model_ready")

con.execute("DROP TABLE IF EXISTS crime_type_model_ready;")

con.execute(f"""
CREATE TABLE crime_type_model_ready AS
WITH top_crime_types AS (
    SELECT primary_type
    FROM crimes_clean
    WHERE crime_date <= DATE '{training_end_date}'
      AND primary_type IS NOT NULL
    GROUP BY primary_type
    ORDER BY COUNT(*) DESC
    LIMIT 15
),
calendar AS (
    SELECT *
    FROM generate_series(
        DATE '2001-01-01',
        DATE '{training_end_date}',
        INTERVAL 1 DAY
    ) AS t(date)
),
grid AS (
    SELECT
        c.date,
        t.primary_type
    FROM calendar c
    CROSS JOIN top_crime_types t
),
daily AS (
    SELECT
        date,
        primary_type,
        crime_count
    FROM crime_type_daily_counts
    WHERE date <= DATE '{training_end_date}'
)
SELECT
    g.date,
    g.primary_type,
    COALESCE(d.crime_count, 0) AS crime_count,

    EXTRACT(day FROM g.date) AS day,
    EXTRACT(month FROM g.date) AS month,
    EXTRACT(year FROM g.date) AS year,
    EXTRACT(dow FROM g.date) AS day_of_week,
    EXTRACT(week FROM g.date) AS week_of_year,

    CASE
        WHEN EXTRACT(dow FROM g.date) IN (0, 6) THEN TRUE
        ELSE FALSE
    END AS is_weekend

FROM grid g
LEFT JOIN daily d
    ON g.date = d.date
   AND g.primary_type = d.primary_type
ORDER BY g.date, g.primary_type;
""")

count = con.execute("SELECT COUNT(*) FROM crime_type_model_ready;").fetchone()[0]
print(f"crime_type_model_ready rows: {count:,}")

# --------------------------------------------------
# 5. Create district + crime type model-ready table for XGBoost
# --------------------------------------------------

print("\n[5/7] Creating district_crime_type_model_ready")

con.execute("DROP TABLE IF EXISTS district_crime_type_model_ready;")

con.execute(f"""
CREATE TABLE district_crime_type_model_ready AS
WITH valid_districts AS (
    SELECT district
    FROM (
        VALUES
        (1), (2), (3), (4), (5), (6), (7), (8), (9), (10),
        (11), (12), (14), (15), (16), (17), (18), (19), (20),
        (22), (24), (25)
    ) AS t(district)
),
top_crime_types AS (
    SELECT primary_type
    FROM crimes_clean
    WHERE crime_date <= DATE '{training_end_date}'
      AND primary_type IS NOT NULL
    GROUP BY primary_type
    ORDER BY COUNT(*) DESC
    LIMIT 15
),
calendar AS (
    SELECT *
    FROM generate_series(
        DATE '2001-01-01',
        DATE '{training_end_date}',
        INTERVAL 1 DAY
    ) AS t(date)
),
grid AS (
    SELECT
        c.date,
        d.district,
        t.primary_type
    FROM calendar c
    CROSS JOIN valid_districts d
    CROSS JOIN top_crime_types t
),
daily AS (
    SELECT
        date,
        district,
        primary_type,
        crime_count
    FROM district_crime_type_daily_counts
    WHERE date <= DATE '{training_end_date}'
)
SELECT
    g.date,
    g.district,
    g.primary_type,
    COALESCE(d.crime_count, 0) AS crime_count,

    EXTRACT(day FROM g.date) AS day,
    EXTRACT(month FROM g.date) AS month,
    EXTRACT(year FROM g.date) AS year,
    EXTRACT(dow FROM g.date) AS day_of_week,
    EXTRACT(week FROM g.date) AS week_of_year,

    CASE
        WHEN EXTRACT(dow FROM g.date) IN (0, 6) THEN TRUE
        ELSE FALSE
    END AS is_weekend

FROM grid g
LEFT JOIN daily d
    ON g.date = d.date
   AND g.district = d.district
   AND g.primary_type = d.primary_type
ORDER BY g.date, g.district, g.primary_type;
""")

count = con.execute("SELECT COUNT(*) FROM district_crime_type_model_ready;").fetchone()[0]
print(f"district_crime_type_model_ready rows: {count:,}")

# --------------------------------------------------
# 6. Add lag and rolling features for XGBoost
# --------------------------------------------------

print("\n[6/7] Creating xgboost_features_daily")

con.execute("DROP TABLE IF EXISTS xgboost_features_daily;")

con.execute("""
CREATE TABLE xgboost_features_daily AS
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
    is_weekend,

    LAG(crime_count, 1) OVER (
        PARTITION BY district, primary_type
        ORDER BY date
    ) AS lag_1,

    LAG(crime_count, 7) OVER (
        PARTITION BY district, primary_type
        ORDER BY date
    ) AS lag_7,

    LAG(crime_count, 14) OVER (
        PARTITION BY district, primary_type
        ORDER BY date
    ) AS lag_14,

    LAG(crime_count, 30) OVER (
        PARTITION BY district, primary_type
        ORDER BY date
    ) AS lag_30,

    AVG(crime_count) OVER (
        PARTITION BY district, primary_type
        ORDER BY date
        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS rolling_mean_7,

    AVG(crime_count) OVER (
        PARTITION BY district, primary_type
        ORDER BY date
        ROWS BETWEEN 30 PRECEDING AND 1 PRECEDING
    ) AS rolling_mean_30,

    STDDEV(crime_count) OVER (
        PARTITION BY district, primary_type
        ORDER BY date
        ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING
    ) AS rolling_std_7

FROM district_crime_type_model_ready;
""")

con.execute("""
DELETE FROM xgboost_features_daily
WHERE lag_30 IS NULL
   OR rolling_mean_30 IS NULL;
""")

count = con.execute("SELECT COUNT(*) FROM xgboost_features_daily;").fetchone()[0]
print(f"xgboost_features_daily rows: {count:,}")

# --------------------------------------------------
# 7. Cleaner high-risk time periods with minimum support
# --------------------------------------------------

print("\n[7/7] Creating high_risk_time_periods_v2")

con.execute("DROP TABLE IF EXISTS high_risk_time_periods_v2;")

con.execute(f"""
CREATE TABLE high_risk_time_periods_v2 AS
WITH valid_districts AS (
    SELECT district
    FROM (
        VALUES
        (1), (2), (3), (4), (5), (6), (7), (8), (9), (10),
        (11), (12), (14), (15), (16), (17), (18), (19), (20),
        (22), (24), (25)
    ) AS t(district)
),
top_crime_types AS (
    SELECT primary_type
    FROM crimes_clean
    WHERE crime_date <= DATE '{training_end_date}'
      AND primary_type IS NOT NULL
    GROUP BY primary_type
    ORDER BY COUNT(*) DESC
    LIMIT 15
),
slot_counts AS (
    SELECT
        c.district,
        c.primary_type,
        c.day_of_week,
        c.hour,
        COUNT(*) AS crime_count
    FROM crimes_clean c
    JOIN valid_districts d
        ON c.district = d.district
    JOIN top_crime_types t
        ON c.primary_type = t.primary_type
    WHERE c.crime_date <= DATE '{training_end_date}'
      AND c.hour IS NOT NULL
      AND c.day_of_week IS NOT NULL
    GROUP BY
        c.district,
        c.primary_type,
        c.day_of_week,
        c.hour
),
group_stats AS (
    SELECT
        district,
        primary_type,
        SUM(crime_count) AS group_total,
        AVG(crime_count) AS avg_slot_count
    FROM slot_counts
    GROUP BY district, primary_type
),
risk_calc AS (
    SELECT
        s.district,
        s.primary_type,
        s.day_of_week,
        s.hour,
        s.crime_count,
        g.group_total,

        ROUND(
            100.0 * s.crime_count / NULLIF(g.group_total, 0),
            2
        ) AS share_of_group_percent,

        ROUND(
            100.0 * (s.crime_count - g.avg_slot_count)
            / NULLIF(g.avg_slot_count, 0),
            2
        ) AS risk_percent_vs_group_avg

    FROM slot_counts s
    JOIN group_stats g
        ON s.district = g.district
       AND s.primary_type = g.primary_type
    WHERE g.group_total >= 1000
      AND s.crime_count >= 20
)
SELECT
    *,
    CASE
        WHEN risk_percent_vs_group_avg >= 75 THEN 'Very High'
        WHEN risk_percent_vs_group_avg >= 40 THEN 'High'
        WHEN risk_percent_vs_group_avg >= 15 THEN 'Moderate'
        ELSE 'Normal'
    END AS risk_level
FROM risk_calc
ORDER BY risk_percent_vs_group_avg DESC;
""")

count = con.execute("SELECT COUNT(*) FROM high_risk_time_periods_v2;").fetchone()[0]
print(f"high_risk_time_periods_v2 rows: {count:,}")

# --------------------------------------------------
# 8. Better district risk summary for map
# --------------------------------------------------
# Compare latest 30 days to previous 365-day baseline, not all 20 years.
# This avoids making every district look negative because crime volume declined over time.

print("\n[Extra] Creating district_risk_summary_v2 for Chicago map")

con.execute("DROP TABLE IF EXISTS district_risk_summary_v2;")

con.execute(f"""
CREATE TABLE district_risk_summary_v2 AS
WITH valid_districts AS (
    SELECT district
    FROM (
        VALUES
        (1), (2), (3), (4), (5), (6), (7), (8), (9), (10),
        (11), (12), (14), (15), (16), (17), (18), (19), (20),
        (22), (24), (25)
    ) AS t(district)
),
daily AS (
    SELECT
        d.date,
        d.district,
        d.crime_count
    FROM district_model_ready d
    JOIN valid_districts v
        ON d.district = v.district
    WHERE d.date <= DATE '{training_end_date}'
),
recent_30 AS (
    SELECT
        district,
        SUM(crime_count) AS recent_30_day_count
    FROM daily
    WHERE date > DATE '{training_end_date}' - INTERVAL 30 DAY
      AND date <= DATE '{training_end_date}'
    GROUP BY district
),
previous_365_baseline AS (
    SELECT
        district,
        AVG(crime_count) * 30 AS expected_30_day_count
    FROM daily
    WHERE date > DATE '{training_end_date}' - INTERVAL 395 DAY
      AND date <= DATE '{training_end_date}' - INTERVAL 30 DAY
    GROUP BY district
)
SELECT
    r.district,
    r.recent_30_day_count,
    ROUND(b.expected_30_day_count, 2) AS expected_30_day_count,

    ROUND(
        100.0 * (r.recent_30_day_count - b.expected_30_day_count)
        / NULLIF(b.expected_30_day_count, 0),
        2
    ) AS risk_percent_vs_recent_baseline,

    CASE
        WHEN 100.0 * (r.recent_30_day_count - b.expected_30_day_count) / NULLIF(b.expected_30_day_count, 0) >= 25 THEN 'Very High'
        WHEN 100.0 * (r.recent_30_day_count - b.expected_30_day_count) / NULLIF(b.expected_30_day_count, 0) >= 10 THEN 'High'
        WHEN 100.0 * (r.recent_30_day_count - b.expected_30_day_count) / NULLIF(b.expected_30_day_count, 0) >= 0 THEN 'Moderate'
        ELSE 'Low'
    END AS risk_level

FROM recent_30 r
JOIN previous_365_baseline b
    ON r.district = b.district
ORDER BY risk_percent_vs_recent_baseline DESC;
""")

count = con.execute("SELECT COUNT(*) FROM district_risk_summary_v2;").fetchone()[0]
print(f"district_risk_summary_v2 rows: {count:,}")

# --------------------------------------------------
# Export new model-ready tables
# --------------------------------------------------

print("\nExporting model-ready tables to Parquet...")

exports = {
    "citywide_model_ready": "citywide_model_ready.parquet",
    "district_model_ready": "district_model_ready.parquet",
    "crime_type_model_ready": "crime_type_model_ready.parquet",
    "district_crime_type_model_ready": "district_crime_type_model_ready.parquet",
    "xgboost_features_daily": "xgboost_features_daily.parquet",
    "high_risk_time_periods_v2": "high_risk_time_periods_v2.parquet",
    "district_risk_summary_v2": "district_risk_summary_v2.parquet",
}

for table, filename in exports.items():
    output_path = (PROCESSED_DIR / filename).resolve().as_posix()
    con.execute(f"COPY {table} TO '{output_path}' (FORMAT PARQUET);")
    print(f"Exported: {filename}")

# --------------------------------------------------
# Preview outputs
# --------------------------------------------------

print("\nPreview: citywide_model_ready latest rows")
rows = con.execute("""
SELECT *
FROM citywide_model_ready
ORDER BY date DESC
LIMIT 10;
""").fetchall()

for row in rows:
    print(row)

print("\nPreview: high_risk_time_periods_v2")
rows = con.execute("""
SELECT
    district,
    primary_type,
    day_of_week,
    hour,
    crime_count,
    share_of_group_percent,
    risk_percent_vs_group_avg,
    risk_level
FROM high_risk_time_periods_v2
LIMIT 20;
""").fetchall()

for row in rows:
    print(row)

print("\nPreview: district_risk_summary_v2")
rows = con.execute("""
SELECT *
FROM district_risk_summary_v2
ORDER BY risk_percent_vs_recent_baseline DESC;
""").fetchall()

for row in rows:
    print(row)

con.close()

elapsed = (time.time() - start_time) / 60

print("\n" + "=" * 80)
print("MODEL-READY TABLE BUILD COMPLETE")
print("=" * 80)
print(f"Runtime: {elapsed:.2f} minutes")
print(f"Training end date: {training_end_date}")
print(f"Processed output folder: {PROCESSED_DIR}")
print("=" * 80)
