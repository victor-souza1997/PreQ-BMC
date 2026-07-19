# Installation Guide

## Minimal Requirements
- Python 3.10 or newer.
- ESBMC on your system PATH or installed locally.
- GCC or another configured compiler when C backend compilation is enabled.
- Optional: Gurobi and gurobipy (for reference runs).

## Basic Editable Install
To install the preqbmc command and lightweight Python package metadata, run the following from the repository root:
```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```
### Optional Dependencies
```bash
pip install -e '.[full]'
```

Smaller optional groups are also available:

- `pip install -e '.[cbc]'`
- `pip install -e '.[gurobi]'`
- `pip install -e '.[plots]'`
- `pip install -e '.[dev]'`

## ESBMC Installation

ESBMC is the bounded model checker used to verify generated C harnesses.

Option 1: Repository-Local Install (Recommended)
```bash 
preqbmc install-esbmc
preqbmc verify-environment
```

This downloads the latest matching ESBMC GitHub release and exposes it under `.local/bin/esbmc.`

Option 2: System Installation
If you prefer a system installation, install ESBMC separately and ensure it is on your `PATH`:
```bash
esbmc --version
preqbmc verify-environment
```

## Solvers Configuration

- CBC: Installed via `pip install -e '.[cbc]'`. This is the default license-free MILP backend for the active robustness pipeline.
- Gurobi: Used only for reference runs selected with `--solver gurobi`. It requires a valid Gurobi installation and an importable `gurobipy` Python package. Do not commit license files or WLS credentials to version control.