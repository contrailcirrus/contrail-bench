# ContrailBench

**ContrailBench** is an analysis framework for evaluating contrail forecast models.

This repository contains public code releases to accompany ContrailBench reports. To view the latest ContrailBench reports, see [bench.contrails.org](https://bench.contrails.org). For background information on ContrailBench, see posts at [notebook.contrails.org](https://notebook.contrails.org).

As ContrailBench matures, we anticipate releasing public software for efficiently curating evaluation datasets and running forecast benchmarks. As of the current release, however, the primary purpose of this repository is transparency.

## Code Releases

Code releases to accompany ContrailBench reports are available in the [reports](reports) subdirectory. See individual subdirectories for information about code releases.

## Reports

ContrailBench reports are built from sources in the [docs](docs) subdirectory.

## Running example notebooks

To set up a virtual environment for running example notebooks provided alongside ContrailBench reports, install [uv](https://docs.astral.sh/uv/) and run
```bash
make venv
source .venv/bin/activate # macOS/Linux
# .venv\Scripts\Activate.ps1 # Window PowerShell
# .venv\Scripts\activate.bat # Windows CMD
make install
```

With the virtual environment active, run
```bash
jupyter lab
```
and navigate to [docs/notebooks](docs/notebooks) to open and run notebooks.
