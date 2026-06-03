# Native C++ Backend Build

This repo now includes an initial C++ backend scaffold for 1D solver acceleration.

## What is implemented now

- `backwater_native` Python extension via `pybind11`.
- Native `solve_banded_full(ab, rhs)` for the pentadiagonal linear solve path.
- Native `assemble_system_core(...)` for the unsteady coefficient matrix/RHS arithmetic core.
- Native `adaptive_damping_scale(...)` for Newton update damping.
- Python solver keeps full compatibility and falls back automatically.

## Enable native solver at runtime

Set this environment variable before launching QGIS/plugin runtime:

```bash
export BACKWATER_USE_CPP_SOLVER=1
```

If the extension is not present or throws, solver falls back to SciPy/NumPy.

## Local build (Linux)

1. Install build prerequisites:

```bash
python3 -m pip install --user pybind11
sudo apt-get install -y cmake build-essential python3-dev
```

2. Configure and build from plugin root:

```bash
cmake -S . -B build
cmake --build build -j
```

3. Make module importable for local testing:

```bash
cp build/backwater_native*.so .
```

4. Verify import:

```bash
python3 -c "import backwater_native; print(backwater_native.__doc__)"
```

## Notes

- This is beyond the Week 1 bootstrap stage but still short of full native 1D parity.
- Current benchmark tooling supports `--backend python|native|compare` for side-by-side timing.
- Current native 1D contract is documented in `docs/NATIVE_1D_BACKEND_CONTRACT.md`.
- Next C++ milestones: native node-state derivative prep and a single-step or full-loop `run_unsteady_1d_cpp(...)` binding.
