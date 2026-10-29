#!/bin/bash
#
# Commands for every forecast pipeline. Everything is commented out: uncomment what you want to run.
#
# Every pipeline is resumable (files that exist are skipped) and takes --dry-run to say what it
# would do and where it would write, without doing it. Outputs go to
# gs://contrails-301217-contrail-bench/v2 unless --root says otherwise. Try a job on a day or two
# (--start, --end, --hours) before running a whole period, and run Dataflow jobs one at a time.

# The period (the default of every pipeline, and that of eval-staging)
START="2025-01-01 00:00"
END="2025-12-31 23:00"
PERIOD=(--start "$START" --end "$END")

# --- Forecasts from products (the API picks the run; no leads)
# pipenv run python -m pipelines.preprocess_contrails_org --region global "${PERIOD[@]}" --runner=dataflow
# pipenv run python -m pipelines.preprocess_google --region global "${PERIOD[@]}"   # runs locally

# --- Forecasts from model data. Each valid hour is written at every lead: its shortest lead (1 to 6
# hours) plus each of --lead-offsets (default: 0 to 48 in steps of 6, so leads up to 54 hours).
# Pass --lead-offsets 0,12,24,36 for only the shortest lead and ones 12, 24 and 36 hours longer.
# Try one day first.
# pipenv run python -m pipelines.preprocess_hres --region global "${PERIOD[@]}" --runner=dataflow
# pipenv run python -m pipelines.preprocess_metoffice --region global "${PERIOD[@]}" --runner=dataflow
# pipenv run python -m pipelines.preprocess_ifs50r1 --region global "${PERIOD[@]}" --runner=dataflow   # 00Z only

# --- ERA5 reanalysis
# pipenv run python -m pipelines.preprocess_era5 --region global "${PERIOD[@]}" --runner=dataflow
