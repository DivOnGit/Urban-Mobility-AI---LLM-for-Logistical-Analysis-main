from __future__ import annotations

import math
from functools import lru_cache

import pandas as pd
from langchain_core.tools import tool

from config import settings
from dataset_loader import TaxiDatasetLoader


@lru_cache(maxsize=1)
def _dataset() -> pd.DataFrame:
    loader = TaxiDatasetLoader(settings.data_path)
    return loader.preprocess(loader.load())


@tool
def distance_calculation(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float) -> str:
    """Calculate haversine distance in miles between two latitude/longitude points."""
    radius_miles = 3958.8
    lat1, lon1, lat2, lon2 = map(math.radians, [origin_lat, origin_lon, dest_lat, dest_lon])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    miles = 2 * radius_miles * math.asin(math.sqrt(a))
    return f"Estimated straight-line distance: {miles:.2f} miles."


@tool
def demand_analysis(group_by: str = "pickup_hour") -> str:
    """Analyze taxi demand by a dataset column such as pickup_hour, pickup_day, or pulocationid."""
    df = _dataset()
    if group_by not in df.columns:
        return f"Column '{group_by}' is unavailable. Available columns include: {', '.join(df.columns[:20])}."
    demand = df.groupby(group_by).size().sort_values(ascending=False).head(10)
    return "Top demand groups: " + "; ".join(f"{idx}: {count}" for idx, count in demand.items())


@tool
def pattern_detection(metric: str = "trip_distance") -> str:
    """Detect high-level distribution patterns for numeric taxi trip metrics."""
    df = _dataset()
    if metric not in df.columns:
        return f"Metric '{metric}' is unavailable."
    series = pd.to_numeric(df[metric], errors="coerce").dropna()
    if series.empty:
        return f"Metric '{metric}' has no numeric values."
    return (
        f"{metric} pattern: mean={series.mean():.2f}, median={series.median():.2f}, "
        f"p90={series.quantile(0.90):.2f}, max={series.max():.2f}."
    )


@tool
def trip_statistics() -> str:
    """Return core trip statistics for the loaded NYC taxi dataset."""
    df = _dataset()
    parts = [f"records={len(df)}"]
    for metric in ("trip_distance", "passenger_count", "total_amount"):
        if metric in df.columns:
            values = pd.to_numeric(df[metric], errors="coerce").dropna()
            if not values.empty:
                parts.append(f"{metric}_mean={values.mean():.2f}")
    return ", ".join(parts)


@tool
def time_filtering(hour: int) -> str:
    """Filter demand for a pickup hour from 0 to 23."""
    df = _dataset()
    if "pickup_hour" not in df.columns:
        return "pickup_hour is unavailable in the loaded dataset."
    subset = df[df["pickup_hour"] == hour]
    return f"Records with pickup_hour={hour}: {len(subset)}."


@tool
def location_filtering(location_column: str, location_id: int) -> str:
    """Count trips for a TLC location id column such as pulocationid or dolocationid."""
    df = _dataset()
    column = location_column.lower()
    if column not in df.columns:
        return f"Column '{column}' is unavailable."
    subset = df[pd.to_numeric(df[column], errors="coerce") == location_id]
    return f"Records where {column}={location_id}: {len(subset)}."


URBAN_MOBILITY_TOOLS = [
    distance_calculation,
    demand_analysis,
    pattern_detection,
    trip_statistics,
    time_filtering,
    location_filtering,
]
