"""Stage raw ContrailWatch attributions."""

import aiohttp
import aiolimiter
import asyncio
import datetime
import itertools
import logging
import re
from typing import Any, Iterable

import gcsfs
import pandas as pd
from google.cloud import bigquery, secretmanager

from pycontrails.core import airports
from pycontrails.utils import coroutines

logging.basicConfig(
    level="WARNING",
    format="[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(module)s] [%(funcName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel("INFO")


# Pipeline parameters

#: Time range
TIMES = (datetime.datetime(2024, 1, 1, 0, 0), datetime.datetime(2024, 12, 31, 23, 0))

#: ContrailWatch batch size (flights per query)
CONTRAILWATCH_BATCH_SIZE = 100

#: Limit size of ContrailWatch bursts
CONTRAILWATCH_BURST_LIMIT = 1000

#: Status codes on which to retry bursts of queries
CONTRAILWATCH_RETRY_ON = (500, 503,)

#: Retry backoff (seconds)
CONTRAILWATCH_RETRY_BACKOFF = 60

#: GCP bucket for temporary assets
GCP_TMPDIR = "gs://contrails-301217-contrail-bench/tmp/2025Q1/contrailwatch-raw"


def get_secret(name: str) -> str:
    """Get secret from GCP secret manager.

    Parameters
    ----------
    name : str
        Name of secret    
        
    Returns
    -------
    str
        Value of secret

    """
    """Get secret from GCP secret manager."""
    client = secretmanager.SecretManagerServiceClient()
    project = "contrails-301217"
    full_name = f"projects/{project}/secrets/{name}/versions/latest"
    response = client.access_secret_version(name=full_name)
    return response.payload.data.decode()


def _airport_iata_lookup() -> pd.Series:
    """Get mapping from airport ICAO to airport IATA code.

    Returns
    -------
    pd.Series
        Airport IATA codes indexedy airport ICAO codes.

    """
    df = airports.global_airport_database()
    df = df[["icao_code", "iata_code"]].dropna()
    return df.set_index("icao_code")["iata_code"]


def get_adsb_metadata() -> pd.DataFrame:
    """Get ADSB metadata for constructing ContrailWatch queries.

    Returns
    -------
    pd.DataFrame
        ADSB metadata.

    """
    cache_path = f"{GCP_TMPDIR}/adsb_metadata.pq"
    try:
        return pd.read_parquet(cache_path)
    except FileNotFoundError:
        logger.warning("ADSB metadata not found in GCS. Submitting BQ query...")

    start = TIMES[0] - pd.Timedelta(minutes=30)
    end = TIMES[1] + pd.Timedelta(minutes=30)

    start_ts = f'TIMESTAMP("{start.strftime("%Y-%m-%dT%H:%M:%SZ")}")'
    end_ts = f'TIMESTAMP("{end.strftime("%Y-%m-%dT%H:%M:%SZ")}")'

    query = (
        'SELECT DISTINCT '
        'airline_iata, '
        'flight_number, '
        'departure_airport_icao, '
        'departure_scheduled_time '
        'FROM `contrails-301217.flights_pipeline_prod.spire_flights_raw_prod` '
        'WHERE '
        f'TIMESTAMP_TRUNC(timestamp, SECOND) BETWEEN {start_ts} AND {end_ts} '
        'AND airline_iata IS NOT NULL '
        'AND flight_number IS NOT NULL '
        'AND departure_airport_icao IS NOT NULL '
        'AND departure_scheduled_time IS NOT NULL '
        'ORDER BY '
        'departure_scheduled_time'
    )

    client = bigquery.Client()
    df = client.query(query).to_dataframe()
    
    # get departure airport IATA code or drop flight
    airport_iata_to_icao = _airport_iata_lookup()
    df = df[df["departure_airport_icao"].isin(airport_iata_to_icao.index)]
    df["departure_airport_iata"] = df["departure_airport_icao"].map(airport_iata_to_icao)

    df = df[[
        "airline_iata",
        "flight_number",
        "departure_scheduled_time",
        "departure_airport_iata"
    ]]
    df.to_parquet(cache_path)
    return df


def _query_param(row: pd.Series) -> str | None:
    """Generate ContrailWatch query parameter for a single flight.

    Returns None for query parameters that would result in a 400 Bad Request.

    Parameters
    ----------
    row : pd.Series
        DataFrame row with metadata for a single flight. Must contain ``airline_iata``,
        ``flight_number``, ``departure_airport_iata``, and ``departure_scheduled_time``
        as columns.

    Returns
    -------
    str | None
        ContrailWatch query parameter.

    """
    carrier = row["airline_iata"]
    if len(carrier) != 2:
        return None
    
    # Flight numbers start with an airline prefix, but for some flights
    # (e.g., flights operated by a regional carrier on behalf of a major airline)
    # the airline prefix differs from the airline iata.
    # In these cases we make an educated guess about the flight number
    # by stripping leading non-digit characters.
    if row["flight_number"].startswith(carrier):
        flight_number = row["flight_number"].removeprefix(carrier)
    else:
        flight_number = re.sub(r"^\D+", "", row["flight_number"])
    if not flight_number.isdigit():  # will reject empty strings
        return None
    if len(flight_number) > 4:
        return None
    
    departure_date = row["departure_scheduled_time"].strftime("%Y-%m-%d")

    departure_airport = row["departure_airport_iata"]
    if not departure_airport.isalpha():
        return None
    if len(departure_airport) != 3:
        return None

    return f"carriers/{carrier}/flightNumbers/{flight_number}/departureDates/{departure_date}/departureAirports/{departure_airport}"


def get_query_parameters(meta: pd.DataFrame) -> list[str]:
    """Get list of ContrailWatch API query parameters.

    Parameters
    ----------
    meta : pd.DataFrame
        Flight metadata

    Returns
    -------
    list[str]
        ContrailWatch query parameter for each flight.

    """
    params = meta.apply(_query_param, axis="columns").dropna()
    return params.tolist()


async def _batch_get(
    names: Iterable[str],
    key: str,
    session: aiohttp.ClientSession,
) -> list[dict[str, Any]]:
    """Submit a single batch request to the ContrailWatch API."""
    url = "https://contrails.googleapis.com/v2/attributions:batchGet"
    params = tuple(("names", name) for name in names)
    headers = {"x-goog-api-key": key}
    async with session.get(url, params=params, headers=headers) as response:
        contents = await response.json()

    # only return responses that include attributed contrail segments
    return [att for att in contents["attributions"] if "segments" in att]


async def _submit_burst(
    names: Iterable[str],
    limiter: aiolimiter.AsyncLimiter
) -> list[dict[str, Any]] | None:
    """Submit a single burst of batch requests."""
    async with limiter:
        try:
            key = get_secret("google-contrails-api-key")
            async with aiohttp.ClientSession(raise_for_status=True) as session:
                batches = itertools.batched(names, CONTRAILWATCH_BATCH_SIZE)
                tasks = [_batch_get(batch, key, session) for batch in batches]
                results = await asyncio.gather(*tasks)
            return list(itertools.chain.from_iterable(results))
        
        except aiohttp.ClientResponseError as e:
            if e.status in CONTRAILWATCH_RETRY_ON:
                logger.warning(f"Burst failed ({e.status} {e.message}). Retrying after backoff.")
                await asyncio.sleep(CONTRAILWATCH_RETRY_BACKOFF)
                return None
            raise e


async def get_attributions(params: Iterable[str], limiter: aiolimiter.AsyncLimiter) -> pd.DataFrame:
    """Get DataFrame with attributed flight segments.

    Parameters
    ----------
    params : list[str]
        Parameters for a single burst of queries

    limiter : aiolimiter.AsyncLimiter
        Rate limiter for ContrailWatch bursts

    Returns
    -------
    pd.DataFrame
        Attributed flight segments. Includes the flight's ICAO address and callsign
        and the segment start and end time.
    
    """
    while (result := await _submit_burst(params, limiter)) is None:
        pass

    if len(result) == 0:
        logger.warning("No attributed segments returned from ContrailWatch API")
        return pd.DataFrame(columns=["icao_address", "callsign", "start", "end"])

    start = TIMES[0] - pd.Timedelta(minutes=30)
    end = TIMES[1] + pd.Timedelta(minutes=30)
    
    segment_list = []
    for flight in result:
        icao_address = flight["flightDetails"]["icao24"]
        callsign = flight["flightDetails"]["callSign"]
        
        for segment in flight["segments"]:
            segment_start = pd.to_datetime(segment["startTime"]).tz_localize(None)
            segment_end = pd.to_datetime(segment["endTime"]).tz_localize(None)
            if segment_start > end or segment_end < start:  # no overlap with target time range
                continue
            
            segment_list.append({
                "icao_address": icao_address,
                "callsign": callsign,
                "start": segment_start,
                "end": segment_end
            })

    return pd.DataFrame(segment_list)


async def stage_contrailwatch_raw() -> None:
    """Stage raw ContrailWatch attributions."""

    limiter = aiolimiter.AsyncLimiter(1)  # limit to 1 burst per minute
    meta = get_adsb_metadata()
    
    for date, group in meta.groupby(meta["departure_scheduled_time"].dt.date):

        date_str = date.strftime("%Y-%m-%d")
        sink = f"{GCP_TMPDIR}/{date_str}.pq"
        if gcsfs.GCSFileSystem().exists(sink):
            logger.debug(f"Query results for flights departing {date_str} already exist")
            continue

        params = get_query_parameters(group)
        num_queries = (len(params) + CONTRAILWATCH_BATCH_SIZE - 1) // CONTRAILWATCH_BATCH_SIZE
        logger.info(f"Querying {len(params)} flights ({num_queries} queries) departing {date_str}")

        if num_queries <= CONTRAILWATCH_BURST_LIMIT:
            attributions = await get_attributions(params, limiter)
            attributions.to_parquet(sink)
            continue

        logger.info("Large number of flights requires multiple bursts.")
        bursts = itertools.batched(params, CONTRAILWATCH_BATCH_SIZE * CONTRAILWATCH_BURST_LIMIT)
        tasks = [get_attributions(burst, limiter) for burst in bursts]
        attributions_list = await asyncio.gather(*tasks)
        attributions = pd.concat(attributions_list, axis="index")
        attributions.to_parquet(sink)


def main() -> None:
    """Program entrypoint."""
    coroutines.run(stage_contrailwatch_raw())


if __name__ == "__main__":
    main()
