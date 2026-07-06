---
FOR IMMEDIATE RELEASE
---

# HYDRA2DGPU v1.2 Delivers GPU-Accelerated 2D Flood Modeling with Headless CLI, 1D Pipe Networks, and 10× Faster Mesh Building

**KANSAS CITY, Mo. — July 2026** — The HYDRA team today announced the release
of HYDRA2DGPU v1.2, a major update to the open-source QGIS plugin for
GPU-accelerated shallow water equation modeling. The release represents over
400 commits and more than 560,000 lines of new engineering software, bringing
significant advances in performance, automation, and hydraulic modeling
capability.

## What's New in v1.2

**Headless CLI & Batch Simulation.** Engineers can now run HYDRA simulations
directly from the command line without launching QGIS, enabling automated
batch execution on servers and CI pipelines. A new batch runner supports
parameter sweep expansion across multiple GPKG outputs, making sensitivity
analyses and ensemble forecasting practical for the first time.

**GPU-Accelerated 1D Pipe Networks.** HYDRA v1.2 adds a complete GPU-native
pipe solver supporting EGL, Diffusion Wave, and Fully Dynamic wave modes,
coupled directly to the 2D overland flow solver. Inlet/exit losses follow
HEC-22 standards, and all hydraulic structures (weirs, culverts, gates,
bridges, pumps) are supported in the headless path.

**Threading & Responsiveness.** The simulation pipeline has been fully
re-architected onto background `QThread` workers. Simulations no longer
freeze the QGIS interface — users can pan, zoom, inspect layers, or even
start new analyses while a model runs. Safe cancellation, live progress
signals, and background GeoPackage persistence complete the upgrade.

**10× Faster Mesh Building.** A sorted-vector edge deduplication algorithm
accelerates unstructured mesh construction by 3–5×. Combined with on-device
face-flux redistribution (zero PCIe transfers), atomics-free Green-Gauss
gradients, and incremental line metrics (O(n²) → O(n)), overall simulation
throughput gains are dramatic across the board.

**In-Memory Live Results.** Results stream into the QGIS map canvas in real
time during the solve, without intermediate GPKG writes. Final results are
persisted automatically at run completion, and a new "save max only" mode
conserves disk space by storing only peak values plus terminal state.

**Overhauled User Interface.** A dedicated Run dock, HYDRA toolbar with
keyboard shortcuts, GroupBox-organized parameter panels, comprehensive
tooltips, and session persistence across QGIS restarts deliver a more
professional and navigable experience. The full 18-layer structure schema
now ships as portable QML styles.

**MVP Architecture.** Internally, the codebase has undergone a rigorous
Model-View-Presenter refactoring. View protocols, controller isolation, and
10+ service extractions ensure maintainability and testability going forward.
Over 500 lines of dead code were removed, and all 83 bare `except:` blocks
were eliminated.

## Availability

HYDRA2DGPU v1.2 is open source under the MIT License. Source code,
documentation, and installation instructions are available at:

**https://github.com/aspragueumkc/hydra2dgpu**

### System Requirements

- QGIS 3.28+
- Python 3.12+
- CUDA Toolkit 11.x or 12.x
- NVIDIA GPU with Compute Capability ≥ 7.5

### About HYDRA

HYDRA (Hydrodynamics & Runoff Application) is a CUDA-accelerated 2D shallow
water equation solver integrated as a QGIS plugin. It couples surface
hydrodynamics, 1D urban drainage, hydraulic structures, and rainfall/
infiltration — all within the QGIS map canvas. HYDRA is developed at the
University of Missouri-Kansas City.

---

**Media Contact:**
Aaron Sprague
aspragueumkc@github.com
https://github.com/aspragueumkc/hydra2dgpu
