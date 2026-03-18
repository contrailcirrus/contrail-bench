#!/bin/bash
#
# Run all pipelines

# # Raw data staging
# pipenv run python -m pipelines.stage_contrailwatch_raw

# # Forecast preprocessing
# pipenv run python -m pipelines.preprocess_contrailwatch --runner=dataflow
# pipenv run python -m python -m pipelines.preprocess_google

# # ADSB preprocessing
# pipenv run python -m pipelines.preprocess_adsb --runner=dataflow

# # Observation preprocessing
# pipenv run python -m pipelines.preprocess_iagos --runner=dataflow
# pipenv run python -m pipelines.preprocess_gruan
# pipenv run python -m pipelines.preprocess_contrailwatch

# # Contrails.org cost (global)
# pipenv run python -m pipelines.contrails_org_adsb --runner=dataflow
# 
# # Contrails.org cost (ContrailWatch region)
# pipenv run python -m pipelines.contrails_org_adsb_contrailwatch_region --runner=dataflow
# 
# # Contrails.org hit rate (global)
# pipenv run python -m pipelines.contrails_org_iagos --runner=dataflow
# pipenv run python -m pipelines.contrails_org_gruan --runner=dataflow
# pipenv run python -m pipelines.contrails_org_contrailwatch --runner=dataflow

# # Contrails.org hit range (global, flight-distance-weighted)
# pipenv run python -m pipelines.contrails_org_iagos_flight_distance --runner=dataflow
# pipenv run python -m pipelines.contrails_org_gruan_flight_distance --runner=dataflow
# pipenv run python -m pipelines.contrails_org_contrailwatch_flight_distance --runner=dataflow

# # Contrails.org hit range (ContrailWatch region)
# pipenv run python -m pipelines.contrails_org_iagos_contrailwatch_region --runner=dataflow
# pipenv run python -m pipelines.contrails_org_gruan_contrailwatch_region --runner=dataflow
# pipenv run python -m pipelines.contrails_org_contrailwatch_contrailwatch_region --runner=dataflow

# Google cost (global)
pipenv run python -m pipelines.google_adsb --runner=dataflow

# Google cost (ContrailWatch region)
pipenv run python -m pipelines.google_adsb_contrailwatch_region --runner=dataflow

# Google hit rate (global)
pipenv run python -m pipelines.google_iagos --runner=dataflow
pipenv run python -m pipelines.google_gruan --runner=dataflow
pipenv run python -m pipelines.google_contrailwatch --runner=dataflow

# Google hit range (global, flight-distance-weighted)
pipenv run python -m pipelines.google_iagos_flight_distance --runner=dataflow
pipenv run python -m pipelines.google_gruan_flight_distance --runner=dataflow
pipenv run python -m pipelines.google_contrailwatch_flight_distance --runner=dataflow

# Google hit range (ContrailWatch region)
pipenv run python -m pipelines.google_iagos_contrailwatch_region --runner=dataflow
pipenv run python -m pipelines.google_gruan_contrailwatch_region --runner=dataflow
pipenv run python -m pipelines.google_contrailwatch_contrailwatch_region --runner=dataflow
