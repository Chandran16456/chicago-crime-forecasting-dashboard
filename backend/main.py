from pathlib import Path
from typing import Optional
import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DASHBOARD_DIR = ROOT_DIR / "data" / "processed" / "dashboard"
FRONTEND_DIR = ROOT_DIR / "frontend"


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------

app = FastAPI(
    title="Chicago Crime Forecasting API",
    description=(
        "FastAPI backend for Chicago crime forecasting, district risk, "
        "crime type trends, high-risk time periods, and dashboard frontend."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Static frontend files
# --------------------------------------------------
# This serves:
# frontend/style.css  -> /static/style.css
# frontend/app.js     -> /static/app.js

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def check_file(path: Path):
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"Required file not found: {path.name}. "
                "Run scripts/09_build_dashboard_outputs.py first."
            ),
        )


def read_csv_file(filename: str):
    """
    Safely read CSV and return JSON-safe records.
    Handles NaN, Infinity, -Infinity, NumPy types, and blank values.
    """
    path = DASHBOARD_DIR / filename
    check_file(path)

    df = pd.read_csv(path)

    # Replace infinite values with NaN first
    df = df.replace([np.inf, -np.inf], np.nan)

    # Convert NaN to None
    df = df.where(pd.notnull(df), None)

    records = df.to_dict(orient="records")

    # Extra safety: recursively clean bad float values
    def clean_value(value):
        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                return None
            return value

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            if np.isnan(value) or np.isinf(value):
                return None
            return float(value)

        return value

    cleaned_records = []

    for row in records:
        cleaned_row = {}
        for key, value in row.items():
            cleaned_row[key] = clean_value(value)
        cleaned_records.append(cleaned_row)

    return cleaned_records


def read_json_file(filename: str):
    path = DASHBOARD_DIR / filename
    check_file(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------
# Frontend routes
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Chicago Crime Forecasting API is running",
        "dashboard": "/dashboard",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/dashboard")
def dashboard():
    index_path = FRONTEND_DIR / "index.html"

    if not index_path.exists():
        raise HTTPException(
            status_code=404,
            detail="frontend/index.html not found",
        )

    return FileResponse(index_path)


@app.get("/favicon.ico")
def favicon():
    raise HTTPException(status_code=404, detail="No favicon configured")


# --------------------------------------------------
# API routes
# --------------------------------------------------

@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "root_dir": str(ROOT_DIR),
        "dashboard_dir": str(DASHBOARD_DIR),
        "dashboard_dir_exists": DASHBOARD_DIR.exists(),
        "frontend_dir": str(FRONTEND_DIR),
        "frontend_dir_exists": FRONTEND_DIR.exists(),
    }


@app.get("/api/summary")
def get_dashboard_summary():
    return read_json_file("api_dashboard_summary.json")


@app.get("/api/model-comparison")
def get_model_comparison():
    return read_csv_file("api_model_comparison.csv")


@app.get("/api/citywide/forecast")
def get_citywide_forecast(days: int = Query(30, ge=1, le=30)):
    data = read_csv_file("api_citywide_30day_forecast.csv")
    return data[:days]


@app.get("/api/citywide/history")
def get_citywide_history(limit: Optional[int] = Query(None, ge=30)):
    data = read_csv_file("api_citywide_history.csv")

    if limit:
        return data[-limit:]

    return data


@app.get("/api/district/risk-map")
def get_district_risk_map():
    return read_csv_file("api_district_risk_map.csv")


@app.get("/api/district/history/{district_id}")
def get_district_history(
    district_id: int,
    limit: Optional[int] = Query(365, ge=30),
):
    data = read_csv_file("api_district_history.csv")

    filtered = [
        row for row in data
        if int(row["district"]) == district_id
    ]

    if not filtered:
        raise HTTPException(
            status_code=404,
            detail=f"No district history found for district {district_id}",
        )

    if limit:
        filtered = filtered[-limit:]

    return filtered


@app.get("/api/crime-types/risk")
def get_crime_type_risk():
    return read_csv_file("api_crime_type_risk.csv")


@app.get("/api/crime-types/history/{primary_type}")
def get_crime_type_history(
    primary_type: str,
    limit: Optional[int] = Query(365, ge=30),
):
    data = read_csv_file("api_crime_type_history.csv")

    requested = primary_type.upper().replace("-", " ")

    filtered = [
        row for row in data
        if str(row["primary_type"]).upper() == requested
    ]

    if not filtered:
        raise HTTPException(
            status_code=404,
            detail=f"No crime type history found for: {primary_type}",
        )

    if limit:
        filtered = filtered[-limit:]

    return filtered


@app.get("/api/high-risk-periods")
def get_high_risk_periods(
    district: Optional[int] = None,
    primary_type: Optional[str] = None,
    limit: int = Query(100, ge=10, le=500),
):
    data = read_csv_file("api_high_risk_time_periods.csv")

    if district is not None:
        data = [
            row for row in data
            if int(row["district"]) == district
        ]

    if primary_type is not None:
        requested = primary_type.upper().replace("-", " ")

        data = [
            row for row in data
            if str(row["primary_type"]).upper() == requested
        ]

    return data[:limit]
