"""
Download all 2026-Q1 benchmark parquet files from public GCS and compile into
a single CSV for use in a Dash app.

Columns in output CSV:
  horizontal_buffer         – forecast buffer radius (Contrails.org only; n/a for Google)
  probability_threshold     – probability threshold (Google only; n/a for Contrails.org)
  penalty                   – flight distance in forecast PCR (ratio)
  iagos_hit_rate            – IAGOS hit rate
  gruan_hit_rate            – GRUAN hit rate
  contrailwatch_hit_rate    – ContrailWatch hit rate (CONUS benchmarks only)
  penalty_lo / _hi          – 95 % CI for penalty
  iagos_hit_rate_lo / _hi   – 95 % CI for IAGOS hit rate
  gruan_hit_rate_lo / _hi   – 95 % CI for GRUAN hit rate
  contrailwatch_hit_rate_lo / _hi  (CONUS benchmarks only)
  region                    – "global" or "conus"
  season                    – "annual", "winter", "spring", "summer", "autumn"
  forecast                  – "contrails-org" or "google"
"""

import io
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://storage.googleapis.com/contrailbench-public-data/2026Q1/benchmarks"

BENCHMARKS = [
    # (file_stem,                     region,   season,   forecast)
    ("global-contrails-org",          "global", "annual", "contrails-org"),
    ("global-winter-contrails-org",   "global", "winter", "contrails-org"),
    ("global-spring-contrails-org",   "global", "spring", "contrails-org"),
    ("global-summer-contrails-org",   "global", "summer", "contrails-org"),
    ("global-autumn-contrails-org",   "global", "autumn", "contrails-org"),
    ("conus-contrails-org",           "conus",  "annual", "contrails-org"),
    ("global-google",                 "global", "annual", "google"),
    ("global-winter-google",          "global", "winter", "google"),
    ("global-spring-google",          "global", "spring", "google"),
    ("global-summer-google",          "global", "summer", "google"),
    ("global-autumn-google",          "global", "autumn", "google"),
    ("conus-google",                  "conus",  "annual", "google"),
]


def read_pq(stem: str) -> pd.DataFrame:
    url = f"{BASE}/{stem}.pq"
    print(f"  GET {url}")
    with urllib.request.urlopen(url) as response:
        data = response.read()
    return pd.read_parquet(io.BytesIO(data))


def main() -> None:
    all_dfs: list[pd.DataFrame] = []

    for stem, region, season, forecast in BENCHMARKS:
        print(f"\n[{stem}]")
        df = read_pq(stem)
        df_ci = read_pq(f"{stem}-ci")
        merged = df.join(df_ci)
        if forecast == "contrails-org":
            merged.index.name = "horizontal_buffer"
            merged = merged.reset_index()
            merged["probability_threshold"] = pd.NA
        else:
            merged.index.name = "probability_threshold"
            merged = merged.reset_index()
            merged["horizontal_buffer"] = pd.NA
        merged["region"] = region
        merged["season"] = season
        merged["forecast"] = forecast
        all_dfs.append(merged)

    combined = pd.concat(all_dfs, ignore_index=True)

    # Reorder: metadata columns first
    meta_cols = ["region", "season", "forecast", "horizontal_buffer", "probability_threshold"]
    data_cols = [c for c in combined.columns if c not in meta_cols]
    combined = combined[meta_cols + data_cols]

    out_path = Path(__file__).parent / "benchmarks.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} rows × {len(combined.columns)} columns → {out_path}")
    print(combined.dtypes.to_string())


if __name__ == "__main__":
    main()
