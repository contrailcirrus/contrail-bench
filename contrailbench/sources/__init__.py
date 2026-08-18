"""Registry of forecast sources.

Each source module exposes a `fetch(time, flight_levels, *, extent, **kwargs)`
function computing PCR fields at many flight levels for a single time, from
one underlying met fetch -- the batching that makes this efficient is the
whole reason sources aren't queried through the per-(time, flight_level)
`Dataloader` interface directly (see `contrailbench.data.PCRStoreDataloader`,
which reads the two-stage store these `fetch()` functions populate).
"""

from contrailbench.sources import metoffice

SOURCES = {
    "metoffice": metoffice,
}
