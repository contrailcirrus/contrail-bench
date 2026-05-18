# Contrail Bench - 2026Q1 Report

This directory holds code for generating results presented in the [2026Q1 ContrailBench report](https://bench.contrails.org/reports/2026-Q1/2026-Q1.html).

Code for data preprocessing is organized in a series of [Apache Beam](https://beam.apache.org/) pipelines run using [Google Cloud Dataflow](https://cloud.google.com/products/dataflow). The pipelines are not designed to be runnable without modification (for example, they read from and write to private cloud buckets) but are provided for transparency. Key outputs from pipelines are provided alongside the ContrailBench release as [public evaluation datasets](https://bench.contrails.org/datasets.html). See [Data Preprocessing](#data-preprocessing) for details.

Code for producing results from preprocessed data is provided in [analysis.ipynb](). This notebook is backed entirely by public data. See [Instructions for Reproducing Results](#instructions-for-reproducing-results) for details on running the notebook.

## Instructions for reproducing results

### Environment setup

Dependencies for code supporting the 2026Q1 report are managed using [pipenv](https://pipenv.pypa.io/en/latest/). With pipenv installed, run
```
pipenv install
```
to install dependencies and
```
pipenv run jupyter lab
```
to open the notebook with Jupyter.

### Running

Preprocessed data used in the above notebook is available for download from a public Google Cloud Storage bucket at [gs://contrailbench-public-data/2026Q1](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1). Commands for downloading preprocessed data using the [gcloud CLI](https://cloud.google.com/cli) are included in the notebook.

By default, the notebook performs uncertainty analysis on preprocessed data using a large number of bootstrap iterations (10,000 resamplings). Runtime can be reduced (possibly at the cost of unconverged confidence intervals) by reducing the number of bootstrap iterations. Alternatively, notebook sections including final processing and bootstrapping can be skipped and final processed data (including confidence intervals) can be read directory from Google Cloud Storage.

Lines that write final processed data and confidence intervals to Google Cloud Storage bucket must remain disabled to avoid permission errors.

## Data preprocessing

With a small number of exceptions (noted below), data preprocessing pipelines are designed to be run using [Google Cloud Dataflow](https://cloud.googe.com/produces/dataflow). Commands for building and deploying a Docker image for use in Dataflow are included in the [Makefile](Makefile).

### Pipelines for forecast preprocessing

Forecasts used in the 2026Q1 report are [available through public APIs](https://bench.contrails.org/forecasts.html).

| File | Purpose | Uses Dataflow |
| -- | --- | --- |
| preprocess_contrails_org.py | Preprocess Contrails.org forecasts | ✅ |
| preprocess_google.py | Preprocess Google forecasts | ⛔|

### Pipelines for observation preprocessing

Final outputs from observation preprocessing pipelines are provided as [public evaluation datasets](https://bench.contrails.org/datasets.html).

| File | Purpose | Uses Dataflow | Public Dataset |
| --- | --- | --- | --- |
| stage_contrailwatch_raw.py | Stage raw ContrailWatch data | ⛔ | |
| preprocess_contrailwatch | Generate ContrailWatch evaluation dataset | ✅ | [gs://contrailbench-public-data/2026Q1/contrailwatch](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/contrailwatch) |
| preprocess_gruan | Generate GRUAN evaluation dataset | ✅ | [gs://contrailbench-public-data/2026Q1/gruan](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/contrailwatch) |
| preprocess_iagos | Generate IAGOS evaluation dataset | ✅ | [gs://contrailbench-public-data/2026Q1/iagos](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/iagos) |

### Pipelines for benchmarking

Final outputs from benchmarking pipelines can be used to [reproduce ContrailBench results](#instructions-for-reproducing-results).

**Contrails.org forecast**

| File | Purpose | Uses Dataflow | Public Dataset |
| --- | --- | --- | --- |
| contrails_org_adsb.py | Compute flight distance penalty for global benchmarks | ✅ | [gs://contrailbench-public-data/2026Q1/contrails-org-adsb](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/contrails-org-adsb)
| contrails_org_gruan.py | Compute GRUAN hit rate for global benchmarks | ✅ | [gs://contrailbench-public-data/2026Q1/contrails-org-gruan](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/contrails-org-iagos)
| contrails_org_iagos.py | Compute IAGOS hit rate for global benchmarks | ✅ | [gs://contrailbench-public-data/2026Q1/contrails-org-iagos](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/contrails-org-gruan)
| contrails_org_adsb_contrailwatch_region.py | Compute flight distance penalty for CONUS benchmark | ✅ | [gs://contrailbench-public-data/2026Q1/contrails-org-adsb](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/contrails-org-adsb-contrailwatch-region)
| contrails_org_gruan_contrailwatch_region.py | Compute GRUAN hit rate for CONUS benchmark | ✅ | [gs://contrailbench-public-data/2026Q1/contrails-org-gruan-contrailwatch-region](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/contrails-org-gruan-contrailwatch-region)
| contrails_org_iagos_contrailwatch_region.py | Compute IAGOS hit rate for CONUS benchmark | ✅ | [gs://contrailbench-public-data/2026Q1/contrails-org-iagos-contrailwatch-region](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/contrails-org-iagos-contrailwatch-region)
| contrails_org_contrailwatch_contrailwatch_region.py | Compute ContrailWatch hit rate for CONUS benchmark | ✅ | [gs://contrailbench-public-data/2026Q1/contrails-org-contrailwatch-contrailwatch-region](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/contrails-org-contrailwatch-contrailwatch-region)

**Google forecast**

| File | Purpose | Uses Dataflow | Public Dataset |
| --- | --- | --- | --- |
| google_adsb.py | Compute flight distance penalty for global benchmarks | ✅ | [gs://contrailbench-public-data/2026Q1/google-adsb](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/google-adsb)
| google_gruan.py | Compute GRUAN hit rate for global benchmarks | ✅ | [gs://contrailbench-public-data/2026Q1/google-gruan](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/google-iagos)
| google_iagos.py | Compute IAGOS hit rate for global benchmarks | ✅ | [gs://contrailbench-public-data/2026Q1/google-iagos](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/google-gruan)
| google_adsb_contrailwatch_region.py | Compute flight distance penalty for CONUS benchmark | ✅ | [gs://contrailbench-public-data/2026Q1/google-adsb](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/google-adsb-contrailwatch-region)
| google_gruan_contrailwatch_region.py | Compute GRUAN hit rate for CONUS benchmark | ✅ | [gs://contrailbench-public-data/2026Q1/google-gruan-contrailwatch-region](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/google-gruan-contrailwatch-region)
| google_iagos_contrailwatch_region.py | Compute IAGOS hit rate for CONUS benchmark | ✅ | [gs://contrailbench-public-data/2026Q1/google-iagos-contrailwatch-region](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/google-iagos-contrailwatch-region)
| google_contrailwatch_contrailwatch_region.py | Compute ContrailWatch hit rate for CONUS benchmark | ✅ | [gs://contrailbench-public-data/2026Q1/google-contrailwatch-contrailwatch-region](https://console.cloud.google.com/storage/browser/contrailbench-public-data/2026Q1/google-contrailwatch-contrailwatch-region)

### Miscellaneous pipelines

| File | Purpose | Uses Dataflow |
| --- | --- | --- |
| count_contrails_org.py | Count number of PCR and total cells in Contrails.org forecast | ✅ |
| count_adsb.py | Count number of non-empty cells in ADS-B evaluation dataset | ✅ |
| count_gruan.py | Count number of PCR and total cells in GRUAN evaluation dataset | ✅ |
| count_contrailwatch.py | Count number of PCR cells in ContrailWatch evaluation dataset | ✅ |
| count_contrailwatch.py | Count number of PCR cells in ContrailWatch evaluation dataset | ✅ |
| iagos_statistics.py | Count cells and flight distances in IAGOS evaluation dataset | ✅ |
