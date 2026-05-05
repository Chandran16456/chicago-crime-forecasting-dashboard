from pathlib import Path
import duckdb
import os
import time

# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT_DIR / "data" / "raw" / "chicago_crime.txt"
DB_PATH = ROOT_DIR / "db" / "chicago_crime.duckdb"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# DuckDB likes forward slashes, even on Windows
RAW_FILE_SQL = RAW_FILE.resolve().as_posix()
CITYWIDE_PARQUET = (PROCESSED_DIR / "citywide_daily_counts.parquet").resolve().as_posix()
DISTRICT_PARQUET = (PROCESSED_DIR / "district_daily_counts.parquet").resolve().as_posix()
CRIME_TYPE_PARQUET = (PROCESSED_DIR / "crime_type_daily_counts.parquet").resolve().as_posix()
DAILY_FULL_PARQUET = (PROCESSED_DIR / "daily_crime_counts.parquet").resolve().as_posix()
RISK_PERIODS_PARQUET = (PROCESSED_DIR / "high_risk_time_periods.parquet").resolve().as_posix()
DISTRICT_RISK_PARQUET = (PROCESSED_DIR / "district_risk_summary.parquet").resolve().as_posix()

print("=" * 80)
print("CHICAGO CRIME DUCKDB LOADING PIPELINE")
print("=" * 80)

if not RAW_FILE.exists():
    raise FileNotFoundError(f"Raw file not found: {RAW_FILE}")

file_size_gb = RAW_FILE.stat().st_size / (1024 ** 3)
print(f"Raw file found: {RAW_FILE}")
print(f"Raw file size: {file_size_gb:.2f} GB")
print(f"DuckDB database path: {DB_PATH}")

start_time = time.time()

# --------------------------------------------------
# Connect to DuckDB
# --------------------------------------------------

con = duckdb.connect(str(DB_PATH))

threads = max(1, os.cpu_count() or 4)
con.execute(f"PRAGMA threads={threads};")
con.execute("PRAGMA enable_progress_bar;")

print(f"Using {threads} CPU threads")

# --------------------------------------------------
# 1. Load raw CSV/TXT into DuckDB
# --------------------------------------------------
# We load everything as VARCHAR first.
# This avoids import failures caused by messy dates, blanks, or mixed column types.

print("\n[1/7] Loading raw TXT file into DuckDB table: raw_crimes")
print("This may take several minutes for a 1.66GB file...\n")

con.execute("DROP TABLE IF EXISTS raw_crimes;")

con.execute(f"""
CREATE TABLE raw_crimes AS
SELECT *
FROM read_csv_auto(
    '{RAW_FILE_SQL}',
    header = true,
    delim = ',',
    quote = '"',
    escape = '"',
    all_varchar = true,
    ignore_errors = true,
    null_padding = true
);
""")

raw_count = con.execute("SELECT COUNT(*) FROM raw_crimes;").fetchone()[0]
print(f"raw_crimes row count: {raw_count:,}")

print("\nRaw table schema:")
schema_rows = con.execute("DESCRIBE raw_crimes;").fetchall()
for row in schema_rows:
    print(row)

# --------------------------------------------------
# 2. Create cleaned table with useful columns
# --------------------------------------------------

print("\n[2/7] Creating cleaned table: crimes_clean")

con.execute("DROP TABLE IF EXISTS crimes_clean;")

con.execute("""
CREATE TABLE crimes_clean AS
WITH parsed AS (
    SELECT
        *,
        try_strptime("Date", '%m/%d/%Y %I:%M:%S %p') AS crime_datetime,
        try_strptime("Updated On", '%m/%d/%Y %I:%M:%S %p') AS updated_on_datetime
    FROM raw_crimes
)
SELECT
    TRY_CAST(NULLIF(TRIM("ID"), '') AS BIGINT) AS id,
    "Case Number" AS case_number,

    crime_datetime,
    CAST(crime_datetime AS DATE) AS crime_date,

    EXTRACT(hour FROM crime_datetime) AS hour,
    EXTRACT(dow FROM crime_datetime) AS day_of_week,
    EXTRACT(week FROM crime_datetime) AS week_of_year,
    EXTRACT(month FROM crime_datetime) AS month,
    EXTRACT(year FROM crime_datetime) AS year,

    CASE
        WHEN EXTRACT(dow FROM crime_datetime) IN (0, 6) THEN TRUE
        ELSE FALSE
    END AS is_weekend,

    "Block" AS block,
    "IUCR" AS iucr,
    "Primary Type" AS primary_type,
    "Description" AS description,
    "Location Description" AS location_description,

    CASE
        WHEN LOWER(TRIM("Arrest")) = 'true' THEN TRUE
        WHEN LOWER(TRIM("Arrest")) = 'false' THEN FALSE
        ELSE NULL
    END AS arrest,

    CASE
        WHEN LOWER(TRIM("Domestic")) = 'true' THEN TRUE
        WHEN LOWER(TRIM("Domestic")) = 'false' THEN FALSE
        ELSE NULL
    END AS domestic,

    TRY_CAST(NULLIF(TRIM("Beat"), '') AS INTEGER) AS beat,
    TRY_CAST(NULLIF(TRIM("District"), '') AS INTEGER) AS district,
    TRY_CAST(NULLIF(TRIM("Ward"), '') AS INTEGER) AS ward,
    TRY_CAST(NULLIF(TRIM("Community Area"), '') AS INTEGER) AS community_area,

    "FBI Code" AS fbi_code,

    TRY_CAST(NULLIF(TRIM("X Coordinate"), '') AS DOUBLE) AS x_coordinate,
    TRY_CAST(NULLIF(TRIM("Y Coordinate"), '') AS DOUBLE) AS y_coordinate,
    TRY_CAST(NULLIF(TRIM("Latitude"), '') AS DOUBLE) AS latitude,
    TRY_CAST(NULLIF(TRIM("Longitude"), '') AS DOUBLE) AS longitude,

    "Location" AS location,
    updated_on_datetime AS updated_on

FROM parsed
WHERE crime_datetime IS NOT NULL;
""")

clean_count = con.execute("SELECT COUNT(*) FROM crimes_clean;").fetchone()[0]
print(f"crimes_clean row count: {clean_count:,}")

# --------------------------------------------------
# 3. Citywide daily counts
# --------------------------------------------------

print("\n[3/7] Creating citywide daily crime count table")

con.execute("DROP TABLE IF EXISTS citywide_daily_counts;")

con.execute("""
CREATE TABLE citywide_daily_counts AS
SELECT
    crime_date AS date,
    COUNT(*) AS crime_count
FROM crimes_clean
GROUP BY crime_date
ORDER BY crime_date;
""")

citywide_count = con.execute("SELECT COUNT(*) FROM citywide_daily_counts;").fetchone()[0]
print(f"citywide_daily_counts rows: {citywide_count:,}")

# --------------------------------------------------
# 4. District daily counts
# --------------------------------------------------

print("\n[4/7] Creating district daily crime count table")

con.execute("DROP TABLE IF EXISTS district_daily_counts;")

con.execute("""
CREATE TABLE district_daily_counts AS
SELECT
    crime_date AS date,
    district,
    COUNT(*) AS crime_count
FROM crimes_clean
WHERE district IS NOT NULL
GROUP BY crime_date, district
ORDER BY crime_date, district;
""")

district_count = con.execute("SELECT COUNT(*) FROM district_daily_counts;").fetchone()[0]
print(f"district_daily_counts rows: {district_count:,}")

# --------------------------------------------------
# 5. Crime type daily counts
# --------------------------------------------------

print("\n[5/7] Creating crime type daily count table")

con.execute("DROP TABLE IF EXISTS crime_type_daily_counts;")

con.execute("""
CREATE TABLE crime_type_daily_counts AS
SELECT
    crime_date AS date,
    primary_type,
    COUNT(*) AS crime_count
FROM crimes_clean
WHERE primary_type IS NOT NULL
GROUP BY crime_date, primary_type
ORDER BY crime_date, primary_type;
""")

crime_type_count = con.execute("SELECT COUNT(*) FROM crime_type_daily_counts;").fetchone()[0]
print(f"crime_type_daily_counts rows: {crime_type_count:,}")

# --------------------------------------------------
# 6. District + crime type daily counts
# --------------------------------------------------

print("\n[6/7] Creating district + crime type daily count table")

con.execute("DROP TABLE IF EXISTS district_crime_type_daily_counts;")

con.execute("""
CREATE TABLE district_crime_type_daily_counts AS
SELECT
    crime_date AS date,
    district,
    primary_type,
    COUNT(*) AS crime_count
FROM crimes_clean
WHERE district IS NOT NULL
  AND primary_type IS NOT NULL
GROUP BY crime_date, district, primary_type
ORDER BY crime_date, district, primary_type;
""")

full_daily_count = con.execute("SELECT COUNT(*) FROM district_crime_type_daily_counts;").fetchone()[0]
print(f"district_crime_type_daily_counts rows: {full_daily_count:,}")

# --------------------------------------------------
# 7. Historical high-risk time periods
# --------------------------------------------------
# This is not a future ML forecast yet.
# This creates the first historical risk-percentage table.
# Later, XGBoost forecasts will power future district/crime-type risk.

print("\n[7/7] Creating historical high-risk time period table")

con.execute("DROP TABLE IF EXISTS high_risk_time_periods;")

con.execute("""
CREATE TABLE high_risk_time_periods AS
WITH slot_counts AS (
    SELECT
        district,
        primary_type,
        day_of_week,
        hour,
        COUNT(*) AS crime_count
    FROM crimes_clean
    WHERE district IS NOT NULL
      AND primary_type IS NOT NULL
      AND hour IS NOT NULL
      AND day_of_week IS NOT NULL
    GROUP BY district, primary_type, day_of_week, hour
),
group_baseline AS (
    SELECT
        district,
        primary_type,
        AVG(crime_count) AS avg_slot_count
    FROM slot_counts
    GROUP BY district, primary_type
)
SELECT
    s.district,
    s.primary_type,
    s.day_of_week,
    s.hour,
    s.crime_count,

    ROUND(
        100.0 * s.crime_count
        / NULLIF(SUM(s.crime_count) OVER (PARTITION BY s.district, s.primary_type), 0),
        2
    ) AS share_of_group_percent,

    ROUND(
        100.0 * (s.crime_count - b.avg_slot_count)
        / NULLIF(b.avg_slot_count, 0),
        2
    ) AS risk_percent_vs_group_avg

FROM slot_counts s
JOIN group_baseline b
  ON s.district = b.district
 AND s.primary_type = b.primary_type
ORDER BY risk_percent_vs_group_avg DESC;
""")

risk_period_count = con.execute("SELECT COUNT(*) FROM high_risk_time_periods;").fetchone()[0]
print(f"high_risk_time_periods rows: {risk_period_count:,}")

# --------------------------------------------------
# 8. District risk summary for future Chicago map
# --------------------------------------------------
# This compares the latest 30 days to each district's historical 30-day baseline.
# Later, this can be replaced or upgraded with model-based forecast risk.

print("\n[Extra] Creating district risk summary table for map")

con.execute("DROP TABLE IF EXISTS district_risk_summary;")

con.execute("""
CREATE TABLE district_risk_summary AS
WITH max_date AS (
    SELECT MAX(crime_date) AS latest_date
    FROM crimes_clean
),
daily AS (
    SELECT
        district,
        crime_date,
        COUNT(*) AS daily_count
    FROM crimes_clean
    WHERE district IS NOT NULL
    GROUP BY district, crime_date
),
baseline AS (
    SELECT
        district,
        AVG(daily_count) * 30 AS expected_30_day_count
    FROM daily
    GROUP BY district
),
recent AS (
    SELECT
        c.district,
        COUNT(*) AS recent_30_day_count
    FROM crimes_clean c
    CROSS JOIN max_date m
    WHERE c.district IS NOT NULL
      AND c.crime_date > m.latest_date - INTERVAL 30 DAY
    GROUP BY c.district
)
SELECT
    r.district,
    r.recent_30_day_count,
    ROUND(b.expected_30_day_count, 2) AS expected_30_day_count,

    ROUND(
        100.0 * (r.recent_30_day_count - b.expected_30_day_count)
        / NULLIF(b.expected_30_day_count, 0),
        2
    ) AS risk_percent_vs_historical_avg

FROM recent r
JOIN baseline b
  ON r.district = b.district
ORDER BY risk_percent_vs_historical_avg DESC;
""")

district_risk_count = con.execute("SELECT COUNT(*) FROM district_risk_summary;").fetchone()[0]
print(f"district_risk_summary rows: {district_risk_count:,}")

# --------------------------------------------------
# Export processed tables to Parquet
# --------------------------------------------------

print("\nExporting processed tables to Parquet...")

con.execute(f"COPY citywide_daily_counts TO '{CITYWIDE_PARQUET}' (FORMAT PARQUET);")
con.execute(f"COPY district_daily_counts TO '{DISTRICT_PARQUET}' (FORMAT PARQUET);")
con.execute(f"COPY crime_type_daily_counts TO '{CRIME_TYPE_PARQUET}' (FORMAT PARQUET);")
con.execute(f"COPY district_crime_type_daily_counts TO '{DAILY_FULL_PARQUET}' (FORMAT PARQUET);")
con.execute(f"COPY high_risk_time_periods TO '{RISK_PERIODS_PARQUET}' (FORMAT PARQUET);")
con.execute(f"COPY district_risk_summary TO '{DISTRICT_RISK_PARQUET}' (FORMAT PARQUET);")

# --------------------------------------------------
# Quick previews
# --------------------------------------------------

print("\nPreview: citywide_daily_counts")
for row in con.execute("SELECT * FROM citywide_daily_counts LIMIT 10;").fetchall():
    print(row)

print("\nPreview: high_risk_time_periods")
for row in con.execute("SELECT * FROM high_risk_time_periods LIMIT 10;").fetchall():
    print(row)

print("\nPreview: district_risk_summary")
for row in con.execute("SELECT * FROM district_risk_summary LIMIT 10;").fetchall():
    print(row)

con.close()

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60

print("\n" + "=" * 80)
print("LOAD COMPLETE")
print("=" * 80)
print(f"Total runtime: {elapsed_minutes:.2f} minutes")
print(f"DuckDB database saved at: {DB_PATH}")
print(f"Processed files saved at: {PROCESSED_DIR}")
print("=" * 80)
