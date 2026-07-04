# Installation

## Basic Editable Install

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

This installs the `preqbmc` command and the lightweight Python package metadata.

## Optional Dependencies

Install all optional dependencies needed for the full experiment pipeline:

```bash
pip install -e '.[full]'
```

Smaller optional groups are available:

```bash
pip install -e '.[gurobi]'
pip install -e '.[plots]'
pip install -e '.[dev]'
```

No strict version pins are declared in `pyproject.toml` because the article environment may differ across machines. If exact experiment versions are required for archival reproducibility, generate a separate `requirements-lock.txt` from the final experiment environment and state that it reflects that environment.

## ESBMC

Install ESBMC separately and ensure it is on `PATH`:

```bash
esbmc --version
preqbmc verify-environment
```

The cached demo and article verification runs require ESBMC.

## Gurobi

Full MILP preimage synthesis requires a valid Gurobi installation and an importable `gurobipy` Python package.

Cached artifact demos do not require Gurobi:

```bash
preqbmc demo --no-gurobi --output output/demo_run
```

Do not commit Gurobi license files, WLS credentials, or logs containing private solver credentials.
