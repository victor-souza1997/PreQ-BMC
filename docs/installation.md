# Installation Guide

This guide installs the SBSeg artifact in editable mode and prepares the solver tools used by the reproducibility scripts.

## Minimal Requirements

- Python 3.10 or newer.
- GCC or another configured C compiler when C backend compilation is enabled.
- ESBMC for verification runs.
- CBC through `python-mip` for the default open-source MILP backend.
- Optional: Gurobi and `gurobipy` for reference runs with `--solver gurobi`.

## Basic Editable Install

Run from the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

This installs the `preqbmc` command and the lightweight package metadata.

## Optional Dependencies

For the full article pipeline, install all optional dependencies:

```bash
pip install -e '.[full]'
```

Smaller groups are available when a machine only needs part of the artifact:

```bash
pip install -e '.[cbc]'
pip install -e '.[gurobi]'
pip install -e '.[plots]'
pip install -e '.[dev]'
```

CBC is the default license-free MILP backend. Gurobi is only needed for `--solver gurobi` reference runs or for regenerating Gurobi-specific preimage caches.

## ESBMC

ESBMC is the bounded model checker used to verify generated C harnesses.

Recommended repo-local install:

```bash
preqbmc install-esbmc
preqbmc verify-environment
```

The installer downloads the latest matching ESBMC GitHub release asset and creates `.local/bin/esbmc`. The ESBMC runner resolves executables in this order:

1. `PREQBMC_ESBMC`, if set;
2. `.local/bin/esbmc` in this repository;
3. `esbmc` from the system `PATH`.

For an opt-in check-and-install flow:

```bash
preqbmc verify-environment --install-missing-esbmc
preqbmc demo --install-missing-esbmc --no-gurobi --output output/demo_run
```

For a direct checkout where the `preqbmc` console command is not installed yet:

```bash
PYTHONPATH=src python src/scripts/install_esbmc.py
```

System ESBMC installations are also supported:

```bash
esbmc --version
preqbmc verify-environment
```

## Environment Check

After installing dependencies, run:

```bash
preqbmc verify-environment
gcc --version
```

`preqbmc verify-environment` reports:

- Python version and executable;
- resolved ESBMC executable and whether the repo-local copy exists;
- CBC/python-mip availability;
- optional Gurobi/gurobipy availability;
- required and optional Python package availability.

Missing Gurobi is not fatal unless `--solver gurobi` is selected. Missing TensorFlow, h5py, or scikit-learn prevents full benchmark runs, but the report explains which package group to install.

## Security And Licensing Notes

Do not commit:

- `gurobi.lic`;
- `*.lic`;
- WLS access IDs, secrets, or license IDs;
- logs containing private solver credentials.

The repo-local ESBMC download lives under `.local/`, which is ignored by git.
