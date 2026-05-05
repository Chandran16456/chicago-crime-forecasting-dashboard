from pathlib import Path
import duckdb

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "db" / "chicago_crime.duckdb"

print("=" * 80)
print("CHICAGO CRIME DUCKDB VALIDATION")
print("=" * 80)
print(f"Database path: {DB_PATH}")

if not DB_PATH.exists():
    raise FileNotFoundError(f"DuckDB database not found: {DB_PATH}")

con = duckdb.connect(str(DB_PATH))

# --------------------------------------------------
# 1. Show tables
# --------------------------------------------------

print("\n[1] Tables in database:")

tables = con.execute("SHOW TABLES;").fetchall()

for table in tables:
    print("-", table[0])

# --------------------------------------------------
# 2. Row counts
# --------------------------------------------------

print("\n[2] Row counts:")

table_names = [
    "raw_crimes",
    "crimes_clean",
    "citywide_daily_counts",
    "district_daily_counts",
    "crime_type_daily_counts",
    "district_crime_type_daily_counts",
    "high_risk_time_periods",
    "district_risk_summary",
]

for table in table_names:
    try:
        count = con.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
        print(f"{table}: {count:,}")
    except Exception as e:
        print(f"{table}: ERROR - {e}")

# --------------------------------------------------
# 3. Date range
# --------------------------------------------------

print("\n[3] Date range:")

date_range = con.execute("""
SELECT 
    MIN(crime_date) AS min_date,
    MAX(crime_date) AS max_date,
    COUNT(DISTINCT crime_date) AS unique_days
FROM crimes_clean;
""").fetchone()

print("Min date:", date_range[0])
print("Max date:", date_range[1])
print("Unique crime days:", date_range[2])

# --------------------------------------------------
# 4. Missing value checks
# --------------------------------------------------

print("\n[4] Missing value checks:")

missing = con.execute("""
SELECT
    COUNT(*) AS total_rows,
    SUM(CASE WHEN crime_date IS NULL THEN 1 ELSE 0 END) AS missing_date,
    SUM(CASE WHEN district IS NULL THEN 1 ELSE 0 END) AS missing_district,
    SUM(CASE WHEN primary_type IS NULL THEN 1 ELSE 0 END) AS missing_primary_type,
    SUM(CASE WHEN latitude IS NULL THEN 1 ELSE 0 END) AS missing_latitude,
    SUM(CASE WHEN longitude IS NULL THEN 1 ELSE 0 END) AS missing_longitude
FROM crimes_clean;
""").fetchone()

labels = [
    "total_rows",
    "missing_date",
    "missing_district",
    "missing_primary_type",
    "missing_latitude",
    "missing_longitude",
]

for label, value in zip(labels, missing):
    print(f"{label}: {value:,}")

# --------------------------------------------------
# 5. Top crime types
# --------------------------------------------------

print("\n[5] Top 15 crime types:")

rows = con.execute("""
SELECT
    primary_type,
    COUNT(*) AS crime_count
FROM crimes_clean
WHERE primary_type IS NOT NULL
GROUP BY primary_type
ORDER BY crime_count DESC
LIMIT 15;
""").fetchall()

for row in rows:
    print(row)

# --------------------------------------------------
# 6. Top districts
# --------------------------------------------------

print("\n[6] Top districts by total crime:")

rows = con.execute("""
SELECT
    district,
    COUNT(*) AS crime_count
FROM crimes_clean
WHERE district IS NOT NULL
GROUP BY district
ORDER BY crime_count DESC
LIMIT 25;
""").fetchall()

for row in rows:
    print(row)

# --------------------------------------------------
# 7. Latest citywide daily counts
# --------------------------------------------------

print("\n[7] Latest 15 citywide daily counts:")

rows = con.execute("""
SELECT
    date,
    crime_count
FROM citywide_daily_counts
ORDER BY date DESC
LIMIT 15;
""").fetchall()

for row in rows:
    print(row)

# --------------------------------------------------
# 8. Yearly totals
# --------------------------------------------------

print("\n[8] Yearly totals:")

rows = con.execute("""
SELECT
    year,
    COUNT(*) AS crime_count
FROM crimes_clean
WHERE year IS NOT NULL
GROUP BY year
ORDER BY year;
""").fetchall()

for row in rows:
    print(row)

# --------------------------------------------------
# 9. Top historical high-risk time periods
# --------------------------------------------------

print("\n[9] Top high-risk time periods:")

rows = con.execute("""
SELECT
    district,
    primary_type,
    day_of_week,
    hour,
    crime_count,
    share_of_group_percent,
    risk_percent_vs_group_avg
FROM high_risk_time_periods
ORDER BY risk_percent_vs_group_avg DESC
LIMIT 20;
""").fetchall()

for row in rows:
    print(row)

# --------------------------------------------------
# 10. District risk summary for map
# --------------------------------------------------

print("\n[10] District risk summary:")

rows = con.execute("""
SELECT
    district,
    recent_30_day_count,
    expected_30_day_count,
    risk_percent_vs_historical_avg
FROM district_risk_summary
ORDER BY risk_percent_vs_historical_avg DESC;
""").fetchall()

for row in rows:
    print(row)

con.close()

print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
