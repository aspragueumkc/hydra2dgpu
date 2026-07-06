🚀 **HYDRA2DGPU v1.2 is here!** The biggest release yet for our open-source,
GPU-accelerated flood modeling plugin for QGIS.

**What's new:**

⚡ **Headless CLI** — Run simulations from the terminal, no QGIS required.
Perfect for batch runs, servers, and CI pipelines.

🌊 **1D Pipe Networks on GPU** — EGL, Diffusion Wave & Fully Dynamic pipe
solvers coupled to 2D overland flow. Inlet/exit losses per HEC-22.

🧵 **No more freezes** — Full threading re-architecture: simulations run in
background threads while you keep working in QGIS.

🔬 **Live in-memory results** — Watch results stream onto your map during
the solve. No waiting for GPKG writes.

⚙️ **10× faster mesh building** — Sorted-vector edge dedup, atomics-free
gradients, on-device flux redistribution.

🎨 **UI overhaul** — Run dock, keyboard shortcuts, group-box controls,
complete tooltips, session persistence.

🏗️ **MVP architecture** — Cleaner, testable, maintainable. 500+ lines of
dead code removed, zero bare except handlers remain.

413 commits, 344 files changed since v1.1. MIT-licensed, free for everyone.

👇 Repo & docs:
https://github.com/aspragueumkc/hydra2dgpu

#HYDRA2DGPU #QGIS #GPUComputing #FloodModeling #OpenSource #CUDA #HydraulicModeling #CoastalEngineering
