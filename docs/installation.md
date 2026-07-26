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
python -m pip install -e .
```

This installs the `preqbmc` command and the lightweight package metadata.

For the default license-free SBSeg environment, `requirements.txt` installs
the current checkout with the `paper` dependency group:

```bash
python -m pip install -r requirements.txt
```

This curated file lists only the local `paper` extra. Pip resolves its
transitive dependencies. The historical workstation freeze is retained as
`requirements-workstation-lock.txt` for provenance, but it is not recommended
for installation because it includes machine-specific CUDA, Jupyter, Gurobi,
and transitive packages.

Do not replace the local editable install with another repository using the
same `preqbmc` distribution name. That can leave a console script which imports
a different or missing `cli` module.

## Optional Dependencies

For the license-free article pipeline:

```bash
python -m pip install -e '.[paper]'
```

The article benchmarks use the repository's preconverted HDF5 weights, so the
`paper` group installs PyTorch for dataset utilities but does not install ONNX
conversion tooling, explicit CUDA-toolkit packages, or Gurobi.

Smaller groups are available when a machine only needs part of the artifact:

```bash
python -m pip install -e '.[benchmarks]'
python -m pip install -e '.[cbc]'
python -m pip install -e '.[onnx]'
python -m pip install -e '.[gurobi]'
python -m pip install -e '.[plots]'
python -m pip install -e '.[dev]'
```

CBC is the default license-free MILP backend. Gurobi is only needed for
`--solver gurobi` reference runs or for regenerating Gurobi-specific preimage
caches. The `full` group adds Gurobi to all open-source runtime dependencies:

```bash
python -m pip install -e '.[full]'
```

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

## Console Command Troubleshooting

After activation, both the interpreter and console command must come from the
same virtual environment:

```bash
command -v python
command -v preqbmc
python -c "import cli; print(cli.__file__)"
```

For a checkout at `/path/to/PreQ-BMC`, the expected paths are:

```text
/path/to/PreQ-BMC/.venv/bin/python
/path/to/PreQ-BMC/.venv/bin/preqbmc
/path/to/PreQ-BMC/src/cli.py
```

If a shell previously resolved a global or Conda-installed `preqbmc`, refresh
its command cache after activating the virtual environment:

```bash
rehash  # zsh
hash -r # bash
```

## Security And Licensing Notes

Do not commit:

- `gurobi.lic`;
- `*.lic`;
- WLS access IDs, secrets, or license IDs;
- logs containing private solver credentials.

The repo-local ESBMC download lives under `.local/`, which is ignored by git.
