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
pip install -e '.[cbc]'
pip install -e '.[gurobi]'
pip install -e '.[plots]'
pip install -e '.[dev]'
```

No strict version pins are declared in `pyproject.toml` because the article environment may differ across machines. If exact experiment versions are required for archival reproducibility, generate a separate `requirements-lock.txt` from the final experiment environment and state that it reflects that environment.

## ESBMC

Install ESBMC into this repository:

```bash
preqbmc install-esbmc
preqbmc verify-environment
```

The installer downloads the latest ESBMC GitHub release asset matching the current platform and creates a repo-local executable at `.local/bin/esbmc`. The PreQ-BMC ESBMC runner resolves executables in this order:

1. `PREQBMC_ESBMC`, if set;
2. `.local/bin/esbmc` in this repository;
3. `esbmc` from the system `PATH`.

For an opt-in check-and-install flow, use:

```bash
preqbmc verify-environment --install-missing-esbmc
preqbmc demo --install-missing-esbmc --no-gurobi --output output/demo_run
```

For a direct checkout without installing the `preqbmc` console command, use:

```bash
PYTHONPATH=src python src/scripts/install_esbmc.py
```

If you prefer a system installation, install ESBMC separately and ensure it is on `PATH`:

```bash
esbmc --version
preqbmc verify-environment
```

The cached demo and article verification runs require ESBMC.

## CBC

CBC is the default license-free MILP backend for the active robustness pipeline:

```bash
pip install -e '.[cbc]'
```

## Gurobi

Gurobi is optional and is used only for reference runs selected with `--solver gurobi`. It requires a valid Gurobi installation and an importable `gurobipy` Python package.

Cached artifact demos do not require Gurobi:

```bash
preqbmc demo --no-gurobi --output output/demo_run
```

Do not commit Gurobi license files, WLS credentials, or logs containing private solver credentials.
