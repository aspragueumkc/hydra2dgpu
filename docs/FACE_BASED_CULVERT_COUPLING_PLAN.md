# Face-Based Culvert Coupling — Implementation Plan

> **Created**: 2025-06-05  
> **Scope**: Add an optional face-based flux coupling mechanism for culverts on the full GPU path  
> **Priority**: Engineering — preserves strict mass conservation and momentum balance  
> **Applies to**: `StructureType.CULVERT` on the CUDA solver path only  

---

## 1. Problem Statement

### 1.1 Current Cell-Center Source/Sink Approach

The existing coupling mechanism treats culvert flow as a **source/sink at cell centers**. The GPU kernel `swe2d_coupling_structure_source_kernel` computes:

```
source_rate[upstream_cell]  -= Q / A_up     // removal (L/T depth rate)
source_rate[downstream_cell] += Q / A_down   // addition  (L/T depth rate)
```

This is then added to the `external_source_mps` array and consumed by `swe2d_update_kernel` as a depth-rate term:

```
h_new = h_old + dt * (flux_contribution + src) * inv_area
```

### 1.2 Deficiencies

| Issue | Impact |
|---|---|
| **No momentum transfer** | The full discharge Q passes between cells, but only the mass flux (∝ Q) is applied. Momentum flux (∝ Q·u) is entirely lost. A culvert draining a cell with velocity u removes mass but leaves momentum unchanged, artificially increasing the cell velocity u = hu/h. |
| **Non-conservative discretisation** | Source/sink terms are applied **after** the flux-based update, violating the discrete divergence form. The governing shallow-water equations require div(F) at cell faces for mass and momentum; injecting source at cell centers introduces an O(Δx) inconsistency. |
| **Wet/dry front instability** | When the upstream cell is nearly dry (h → h_min), dividing Q by its area amplifies noise. The update kernel's sub-stepping helps, but the fundamental issue is that source terms bypass the positivity-preserving flux machinery. |
| **Area-dependent artefacts** | The depth rate Q/A depends on cell area, creating mesh-dependent mass transfer rates that should be area-independent for a well-posed hydraulic structure. |

### 1.3 Goal

Introduce an **optional face-based flux coupling** for culverts that:

1. Injects mass **and** momentum through the same Riemann-solver pathway used for natural inter-cell fluxes.
2. Preserves **strict mass conservation** — the total mass gained by the downstream cell equals the total mass lost by the upstream cell, exactly, every time step.
3. Preserves **momentum balance** — momentum fluxes are consistent with the transferred mass and the local velocity field.
4. Is **opt-in** and only applies to culverts on the GPU path, leaving the existing source/sink path for weirs, orifices, pumps, and bridges untouched.

---

## 2. Theoretical Foundation

### 2.1 Shallow-Water Equations in Integral Form

The 2D SWE on a control volume Ω with boundary ∂Ω are:

$$\frac{\partial}{\partial t}\int_\Omega \mathbf{U}\,dA + \oint_{\partial\Omega} \mathbf{F}\cdot\mathbf{n}\,d\Gamma = \int_\Omega \mathbf{S}\,dA$$

where:

$$\mathbf{U} = \begin{pmatrix} h \\ hu \\ hv \end{pmatrix}, \quad \mathbf{F} = \begin{pmatrix} hu & hv \\ hu^2 + \frac{1}{2}gh^2 & huv \\ huv & hv^2 + \frac{1}{2}gh^2 \end{pmatrix}$$

For a finite-volume cell $c_i$ with edges $e_1,\ldots,e_m$, the semi-discrete update is:

$$\frac{d\mathbf{U}_i}{dt} = -\frac{1}{A_i}\sum_{k=1}^{m} \mathbf{F}_{e_k}^* \cdot \mathbf{n}_{e_k}\, \ell_{e_k} + \mathbf{S}_i$$

where $\mathbf{F}_{e_k}^*$ is the Riemann flux at edge $e_k$, $\mathbf{n}_{e_k}$ is the outward normal, and $\ell_{e_k}$ is the edge length.

### 2.2 Culvert as a Virtual Edge

A culvert connecting upstream cell $c_u$ to downstream cell $c_d$ can be modelled as a **virtual edge** inserted into the mesh topology. Conceptually, it adds one additional edge $e_s$ to cell $c_u$ (outward normal pointing toward $c_d$) and one edge to cell $c_d$ (outward normal pointing toward $c_u$).

The virtual edge has:
- **Length** $\ell_s$: the width of the culvert barrel (or overflow width for embankment flow)
- **Normal** $\mathbf{n}_s$: unit vector from $c_u$ to $c_d$ (computed from cell centroids)
- **Riemann flux** $\mathbf{F}_s^*$: determined by the culvert hydraulic capacity

### 2.3 Structure Flux Computation

Unlike a natural edge where the Riemann flux is determined by the left/right states and topography, a culvert imposes a **capacity constraint**: the flux cannot exceed the hydraulic capacity $Q_{\max}$ of the conduit.

The face-based culvert flux algorithm:

1. **Compute the culvert discharge** $Q_c$ using the existing HDS-5 or direct solver (same logic as `swe2d_compute_structure_flows_kernel`).

2. **Determine the flow direction** unit normal $\hat{\mathbf{n}}$ from upstream to downstream cell centroids:
   
   $$\hat{\mathbf{n}} = \frac{\mathbf{x}_d - \mathbf{x}_u}{|\mathbf{x}_d - \mathbf{x}_u|}$$

3. **Compute the structure Riemann flux** $\mathbf{F}_s^*=(F_h, F_{hu}, F_{hv})^T$ at the virtual face:

   $$F_h = Q_c$$
   
   (mass flux: total discharge through the culvert)

   $$F_{hu} = Q_c \cdot u_s + \frac{1}{2}g\,h_s^2\,\hat{n}_x$$
   
   $$F_{hv} = Q_c \cdot v_s + \frac{1}{2}g\,h_s^2\,\hat{n}_y$$

   where $(u_s, v_s)$ is the **structure velocity** and $h_s$ is the **structure depth**:

   - **Upstream-weighted velocity**: $u_s = u_u$, $v_s = v_u$ (velocity of the donating cell)
   - **Donor depth**: $h_s = \max(h_u - z_s, 0)$ where $z_s$ is the culvert invert elevation

   The hydrostatic pressure term $\frac{1}{2}g h_s^2 \hat{\mathbf{n}}$ represents the pressure head driving flow through the structure. This is critical for correctly modelling the momentum exchange.

4. **Apply the flux** to both cells using the standard FVM update:

   $$\frac{d\mathbf{U}_u}{dt} \leftarrow \frac{d\mathbf{U}_u}{dt} - \frac{1}{A_u}\mathbf{F}_s^*\,\ell_s$$
   
   $$\frac{d\mathbf{U}_d}{dt} \leftarrow \frac{d\mathbf{U}_d}{dt} + \frac{1}{A_d}\mathbf{F}_s^*\,\ell_s$$

By construction, $\sum_{\text{all cells}} \Delta h = 0$ exactly, because the same $F_h$ is subtracted from $c_u$ and added to $c_d$. Similarly, momentum is conserved because the same $(F_{hu}, F_{hv})$ appears with opposite signs.

### 2.4 Comparison to Source/Sink

| Property | Source/Sink (current) | Face-Based Flux (proposed) |
|---|---|---|
| Mass conservation | Approximate (round-off within sub-step) | **Exact** by discrete telescoping sum |
| Momentum transfer | None — only depth changes | **Full** — $(Q_c \cdot u_s,\, Q_c \cdot v_s)$ transferred |
| Hydrostatic pressure | Absent | $\frac{1}{2}g h_s^2 \hat{\mathbf{n}}$ drives structure momentum |
| Wet/dry robustness | Q/A pathology near h→0 | Same Riemann limiter cascade as natural edges |
| Mesh dependence | Q/A creates area-dependent rates | Much reduced — only $\ell_s/A$ scaling |

### 2.5 Sub-Critical Check and Flow Limiting

For shallow-water well-posedness, the structure flux must not induce negative depth or super-critical transitions that violate the CFL constraint. The following safeguards are applied:

1. **Depth limiter**: After computing $Q_c$, clamp the mass flux so that:
   
   $$Q_c \cdot \frac{\ell_s}{A_u} \cdot \Delta t \leq \alpha\, h_u$$
   
   where $\alpha$ is a safety factor (default 0.5). This prevents the upstream cell from going dry.

2. **Momentum consistency**: If the mass flux is limited, the momentum flux is limited proportionally:
   
   $$Q_c^{\text{lim}} = \min\left(Q_c,\, \frac{\alpha\, h_u\, A_u}{\ell_s\, \Delta t}\right)$$

3. **Edge CFL check**: The structure velocity $c_s = Q_c / (h_s \cdot \ell_s)$ is compared against the CFL-adaptive time step to ensure stability. If $c_s > \text{CFL} \cdot \sqrt{g h_{\max}}$, the flux is reduced.

### 2.6 Relationship to Existing HLLC Solver

The face-based flux does **not** replace the HLLC Riemann solver for natural edges. The HLLC solver continues to handle all inter-cell fluxes. The structure flux is appended after the HLLC flux accumulation in `swe2d_update_kernel`, following the same pattern as the existing `external_source_mps` pathway but with a separate `external_flux_*` accumulator that carries all three conservative variables (h, hu, hv) rather than just the depth rate.

This design preserves the existing solver architecture and allows the face-based flux to be toggled at runtime without affecting other structure types.

---

## 3. Architecture Overview

### 3.1 Data Flow

```mermaid
graph TB
    subgraph "Python — coupling.py"
        A["SWE2DStructuresSoA<br/>(upstream_cell, downstream_cell,<br/>culvert_*, structure_type)"]
        B["SWE2DCouplingController<br/>face_flux_mode='culvert'"]
        C["FaceFluxSoA<br/>(culvert_face_normal,<br/>culvert_face_width,<br/>velocity_donor_cell)"]
    end

    subgraph "pybind11 Bridge"
        D["swe2d_gpu_upload_culvert_face_flux_params()"]
        E["swe2d_gpu_compute_culvert_face_flux()<br/>→ computes Q_c on device"]
        F["swe2d_gpu_apply_culvert_face_flux()<br/>→ atomicAccum into flux accumulators"]
    end

    subgraph "CUDA Device"
        G["swe2d_compute_structure_flows_kernel<br/>(culvert Q computation — existing)"]
        H["swe2d_culvert_face_flux_kernel<br/>(NEW: Q → (Fh, Fhu, Fhv))"]
        I["d_flux_h / d_flux_hu / d_flux_hv<br/>(existing edge-flux accumulators)"]
    end

    A -->|pack_face_flux_soa()| C
    C -->|H2D upload| D
    G -->|compute Q_c| E
    E -->|Q_c in d_structure_flow| H
    H -->|atomicAdd into| I
    I -->|consumed by swe2d_update_kernel| J["State update"]
```

### 3.2 Key Design Decisions

1. **Reuse the existing structure-flow kernel** (`swe2d_compute_structure_flows_kernel`) to compute culvert discharge $Q_c$. No new hydraulic computation is needed — the same HDS-5 / direct solver logic produces the discharge.

2. **Add a separate kernel** (`swe2d_culvert_face_flux_kernel`) that takes the computed $Q_c$ and produces the three-component flux $(F_h, F_{hu}, F_{hv})$ using the face-normal geometry and donor-cell velocity.

3. **Write into the existing flux accumulators** (`d_flux_h`, `d_flux_hu`, `d_flux_hv`) rather than a separate source array. The update kernel already sums these fluxes with `cell_inv_area`, so no change to `swe2d_update_kernel` is required.

4. **Culvert face width** ($\ell_s$) comes from `culvert_span` for box culverts and `diameter` (or circular-equivalent width) for circular culverts. An additional field `face_width_override` in `HydraulicStructureConfig.metadata` allows user control.

5. **Face normal** is computed from cell centroids: $\hat{\mathbf{n}} = (\mathbf{x}_d - \mathbf{x}_u)/|\mathbf{x}_d - \mathbf{x}_u|$. This requires cell centroids to be available on the GPU (already stored as `d_cell_cx`, `d_cell_cy`).

---

## 4. Implementation Plan

### Phase 1: Data Structures (Python)

**File**: `swe2d/runtime/coupling.py`

#### 4.1 New SoA: `SWE2DCulvertFaceFluxSoA`

```python
@dataclass
class SWE2DCulvertFaceFluxSoA:
    """Face-based flux parameters for culvert structures.
    
    Only populated for structures where structure_type == CULVERT
    and the face_flux coupling mode is active.
    """
    # Indices into the full structure arrays (culverts only)
    structure_index: np.ndarray       # [n_culvert_faces] index into SWE2DStructuresSoA

    # Face geometry
    face_nx: np.ndarray              # [n_culvert_faces] unit normal x
    face_ny: np.ndarray              # [n_culvert_faces] unit normal y
    face_width: np.ndarray            # [n_culvert_faces] culvert face width L_s

    # Donor-cell index for velocity extraction
    donor_cell: np.ndarray            # [n_culvert_faces] upstream cell index
    receiver_cell: np.ndarray         # [n_culvert_faces] downstream cell index

    # Invert elevation for depth computation
    invert_elev: np.ndarray           # [n_culvert_faces] culvert invert elevation

    # Depth limiter safety factor
    depth_safety_factor: np.ndarray   # [n_culvert_faces] α, default 0.5
```

#### 4.2 Controller Extension

Add to `SWE2DCouplingController.__init__`:

```python
self.culvert_face_flux_mode: str = "off"  # "off" | "source_sink" | "face_flux"
```

- `"off"` — no culvert coupling at all (structure disabled)
- `"source_sink"` — current behaviour (cell-center source/sink), default
- `"face_flux"` — new face-based flux path

Add `SWE2DCouplingController._face_flux_soa: Optional[SWE2DCulvertFaceFluxSoA]`

Add `SWE2DCouplingController._build_face_flux_soa()`:

- Iterate over `self._structures_cfg` where `structure_type == CULVERT`.
- Compute face normal from cell centroids: `nx = cx_dn - cx_up`, `ny = cy_dn - cy_up`, normalize.
- Determine face width: `culvert_span` for box, `diameter` for circular, or `face_width_override` from metadata.
- Extract `inlet_invert_elev` as the structure invert.
- Pack into `SWE2DCulvertFaceFluxSoA`.

### Phase 2: GPU Kernel (CUDA)

**File**: `cpp/src/swe2d_gpu.cu`

#### 4.3 New Kernel: `swe2d_culvert_face_flux_kernel`

```cuda
__global__ __launch_bounds__(256, 4) void swe2d_culvert_face_flux_kernel(
    int32_t n_culvert_faces,
    // Pre-computed discharge from structure flow kernel
    const double* __restrict__ structure_flow,    // [n_structures] Q_c in model units
    const int32_t* __restrict__ culvert_struct_idx, // [n_culvert_faces] index into structure arrays
    // Face geometry
    const double* __restrict__ face_nx,            // [n_culvert_faces]
    const double* __restrict__ face_ny,            // [n_culvert_faces]
    const double* __restrict__ face_width,          // [n_culvert_faces] L_s in model units
    // Cell topology
    const int32_t* __restrict__ donor_cell,         // [n_culvert_faces]
    const int32_t* __restrict__ receiver_cell,      // [n_culvert_faces]
    // Invert elevation for depth limiting
    const double* __restrict__ invert_elev,         // [n_culvert_faces]
    const double* __restrict__ depth_safety,         // [n_culvert_faces]
    // Cell state
    const double* __restrict__ cell_h,              // [n_cells]
    const double* __restrict__ cell_hu,             // [n_cells]
    const double* __restrict__ cell_hv,             // [n_cells]
    const double* __restrict__ cell_zb,             // [n_cells]
    const double* __restrict__ cell_area,           // [n_cells]
    // Physical constants
    double gravity,
    double dt,
    double h_min,
    // Output: accumulate into existing flux arrays
    double* __restrict__ flux_h,
    double* __restrict__ flux_hu,
    double* __restrict__ flux_hv)
{
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_culvert_faces) return;

    const int32_t si = culvert_struct_idx[i];
    const double Q_c = structure_flow[si];
    if (!isfinite(Q_c) || Q_c == 0.0) return;

    const int32_t cu = donor_cell[i];
    const int32_t cd = receiver_cell[i];
    if (cu < 0 || cd < 0) return;

    // Determine flow direction and donor cell
    const double sign = (Q_c >= 0.0) ? 1.0 : -1.0;
    const int32_t donor  = (sign >= 0.0) ? cu : cd;
    const int32_t receiver = (sign >= 0.0) ? cd : cu;
    if (donor < 0 || receiver < 0) return;

    const double h_donor = cell_h[donor];
    if (h_donor <= h_min) return;  // dry donor → no flux

    const double hu_donor = cell_hu[donor];
    const double hv_donor = cell_hv[donor];
    const double zb_donor = cell_zb[donor];
    const double inv_h = 1.0 / fmax(h_donor, h_min);
    const double u_donor = hu_donor * inv_h;
    const double v_donor = hv_donor * inv_h;

    // Invert elevation for depth limiting
    const double invert = invert_elev[i];
    const double wse_donor = h_donor + zb_donor;
    const double depth_above_invert = fmax(0.0, wse_donor - invert);

    // Face width and normal
    const double L_s = face_width[i];
    const double nx = face_nx[i];
    const double ny = face_ny[i];
    const double alpha = depth_safety[i];

    // ── Depth limiter: prevent drying the donor cell ──
    const double A_donor = fmax(cell_area[donor], 1.0e-12);
    double Q_lim = Q_c;
    const double max_mass_removal = alpha * h_donor * A_donor / fmax(dt, 1.0e-12);
    const double abs_Q_c = fabs(Q_c);
    if (abs_Q_c > max_mass_removal && max_mass_removal > 0.0) {
        Q_lim = sign * max_mass_removal;
    }

    // ── Structure velocity: donor-cell velocity projected onto face normal ──
    const double vel_normal = u_donor * nx + v_donor * ny;

    // ── Hydrostatic pressure at the face: use depth above invert ──
    const double h_s = fmax(depth_above_invert, 0.0);

    // ── Three-component flux ──
    // F_h  = Q             (mass flux, already limited)
    // F_hu = Q * u_donor + 0.5 * g * h_s^2 * nx  (x-momentum flux)
    // F_hv = Q * u_donor + 0.5 * g * h_s^2 * ny  (y-momentum flux)
    const double F_h  = Q_lim;
    const double F_hu = Q_lim * u_donor + 0.5 * gravity * h_s * h_s * nx;
    const double F_hv = Q_lim * v_donor + 0.5 * gravity * h_s * h_s * ny;

    // Scale by face width (structure "edge length")
    const double scale = L_s;
    const double fh  = F_h  * scale;
    const double fhu = F_hu * scale;
    const double fhv = F_hv * scale;

    // ── Accumulate into the same flux arrays used by the Riemann solver ──
    // Convention: positive Q_c transfers from donor → receiver.
    // The existing edge convention in swe2d_update_kernel is:
    //   if (edge_c0[c] == c) → add flux;  else → subtract
    // For structures, we use atomicAdd as a virtual "extra edge".
    // donor cell loses mass:    subtract
    // receiver cell gains mass: add
    atomicAdd(&flux_h[donor],   -fh);
    atomicAdd(&flux_hu[donor],  -fhu);
    atomicAdd(&flux_hv[donor],  -fhv);
    atomicAdd(&flux_h[receiver],  fh);
    atomicAdd(&flux_hu[receiver], fhu);
    atomicAdd(&flux_hv[receiver], fhv);
}
```

#### 4.4 Orchestration in `swe2d_gpu_compute_coupling_full_on_device`

After `swe2d_compute_structure_flows_kernel` and before `swe2d_coupling_structure_source_kernel`, insert a conditional branch:

```cuda
if (face_flux_params && face_flux_params->n_culvert_faces > 0) {
    int grid = (face_flux_params->n_culvert_faces + BLOCK - 1) / BLOCK;
    swe2d_culvert_face_flux_kernel<<<grid, BLOCK, 0, stream>>>(
        face_flux_params->n_culvert_faces,
        sf_ws.d_structure_flow,           // Q_c from structure flow kernel
        face_flux_params->d_culvert_struct_idx,
        face_flux_params->d_face_nx,
        face_flux_params->d_face_ny,
        face_flux_params->d_face_width,
        face_flux_params->d_donor_cell,
        face_flux_params->d_receiver_cell,
        face_flux_params->d_invert_elev,
        face_flux_params->d_depth_safety,
        dev->d_h,
        dev->d_hu,
        dev->d_hv,
        dev->d_cell_zb,
        cpl_ws.d_cell_area,
        sf_ws.gravity,
        dt,           // current time step
        h_min,
        dev->d_flux_h,
        dev->d_flux_hu,
        dev->d_flux_hv);
    CUDA_CHECK(cudaGetLastError());
} else if (n_structures > 0) {
    // Legacy source/sink path (existing behaviour)
    swe2d_coupling_structure_source_kernel<<<grid, BLOCK, 0, stream>>>(...);
}
```

**Important**: When face_flux mode is active for culverts, `swe2d_coupling_structure_source_kernel` must **skip** culvert-type structures to avoid double-counting. The simplest approach is to mask them out of the source kernel (set `structure_flow[i] = 0` for culverts in face-flux mode) before the source kernel runs.

### Phase 3: Header / Device State (CUDA)

**File**: `cpp/src/swe2d_gpu.cuh`

#### 4.5 New Workspace: `CulvertFaceFluxWorkspace`

```cuda
struct CulvertFaceFluxWorkspace {
    bool     params_preloaded = false;
    int32_t  n_culvert_faces = 0;
    int32_t  face_capacity = 0;

    int32_t* d_culvert_struct_idx = nullptr;  // [n_culvert_faces]
    double*  d_face_nx = nullptr;              // [n_culvert_faces]
    double*  d_face_ny = nullptr;              // [n_culvert_faces]
    double*  d_face_width = nullptr;           // [n_culvert_faces]
    int32_t* d_donor_cell = nullptr;           // [n_culvert_faces]
    int32_t* d_receiver_cell = nullptr;        // [n_culvert_faces]
    double*  d_invert_elev = nullptr;          // [n_culvert_faces]
    double*  d_depth_safety = nullptr;         // [n_culvert_faces]

    void destroy() {
        if (d_culvert_struct_idx) { cudaFree(d_culvert_struct_idx); d_culvert_struct_idx = nullptr; }
        if (d_face_nx) { cudaFree(d_face_nx); d_face_nx = nullptr; }
        if (d_face_ny) { cudaFree(d_face_ny); d_face_ny = nullptr; }
        if (d_face_width) { cudaFree(d_face_width); d_face_width = nullptr; }
        if (d_donor_cell) { cudaFree(d_donor_cell); d_donor_cell = nullptr; }
        if (d_receiver_cell) { cudaFree(d_receiver_cell); d_receiver_cell = nullptr; }
        if (d_invert_elev) { cudaFree(d_invert_elev); d_invert_elev = nullptr; }
        if (d_depth_safety) { cudaFree(d_depth_safety); d_depth_safety = nullptr; }
        n_culvert_faces = 0;
        face_capacity = 0;
        params_preloaded = false;
    }
};

// Add to SWE2DDeviceState:
CulvertFaceFluxWorkspace culvert_ff_ws{};
bool use_culvert_face_flux = false;
```

#### 4.6 New pybind11 Bindings

```cpp
void swe2d_gpu_upload_culvert_face_flux_params(
    int32_t n_culvert_faces,
    const int32_t* culvert_struct_idx,
    const double* face_nx,
    const double* face_ny,
    const double* face_width,
    const int32_t* donor_cell,
    const int32_t* receiver_cell,
    const double* invert_elev,
    const double* depth_safety,
    bool use_face_flux);  // toggle

void swe2d_gpu_compute_culvert_face_flux(
    SWE2DDeviceState* dev,
    double dt,
    double h_min,
    int32_t n_culvert_faces);
```

### Phase 4: Python Binding and Controller

**File**: `swe2d/runtime/coupling.py`

#### 4.7 `_build_face_flux_soa()` Method

```python
def _build_face_flux_soa(self) -> Optional[SWE2DCulvertFaceFluxSoA]:
    """Build SoA for face-based culvert flux coupling."""
    if self.structures is None or self.culvert_face_flux_mode != "face_flux":
        return None

    cfg = self.structures.cfg
    culvert_indices = [
        i for i, st in enumerate(cfg.structures)
        if st.structure_type == StructureType.CULVERT and st.enabled
    ]
    if not culvert_indices:
        return None

    n = len(culvert_indices)
    struct_idx = np.array(culvert_indices, dtype=np.int32)
    donor_cell = np.zeros(n, dtype=np.int32)
    receiver_cell = np.zeros(n, dtype=np.int32)
    face_nx = np.zeros(n, dtype=np.float64)
    face_ny = np.zeros(n, dtype=np.float64)
    face_width = np.zeros(n, dtype=np.float64)
    invert_elev = np.zeros(n, dtype=np.float64)
    depth_safety = np.full(n, 0.5, dtype=np.float64)  # default α = 0.5

    if self._cell_cx is None or self._cell_cy is None:
        return None  # need cell centroids

    for j, i in enumerate(culvert_indices):
        st = cfg.structures[i]
        cu = int(st.upstream_cell)
        cd = int(st.downstream_cell)
        if cu < 0 or cd < 0 or cu >= self.n_cells or cd >= self.n_cells:
            continue

        # Face normal from upstream → downstream centroid
        dx = self._cell_cx[cd] - self._cell_cx[cu]
        dy = self._cell_cy[cd] - self._cell_cy[cu]
        length = max(1.0e-12, math.sqrt(dx*dx + dy*dy))
        face_nx[j] = dx / length
        face_ny[j] = dy / length

        # Flow direction: positive = upstream→downstream
        donor_cell[j] = cu
        receiver_cell[j] = cd

        # Face width
        md = st.metadata
        fwo = float(md.get("face_width_override", 0.0) or 0.0)
        if fwo > 0.0:
            face_width[j] = fwo
        elif int(md.get("culvert_shape", "circular").strip().lower() in ("box", "rect", "rectangular")):
            face_width[j] = float(md.get("culvert_span", md.get("width", 1.0)) or 1.0)
        else:
            face_width[j] = float(md.get("diameter", md.get("culvert_rise", 1.0)) or 1.0)

        invert_elev[j] = float(md.get("inlet_invert_elev", st.crest_elev) or st.crest_elev)
        depth_safety[j] = float(md.get("face_flux_depth_safety", 0.5) or 0.5)

    return SWE2DCulvertFaceFluxSoA(
        structure_index=struct_idx,
        face_nx=face_nx,
        face_ny=face_ny,
        face_width=face_width,
        donor_cell=donor_cell,
        receiver_cell=receiver_cell,
        invert_elev=invert_elev,
        depth_safety_factor=depth_safety,
    )
```

#### 4.8 GPU Upload and Invocation in `apply_native_device_sources`

In the method `apply_native_device_sources`, after `swe2d_compute_structure_flows_kernel` and before `swe2d_coupling_structure_source_kernel`:

```python
# ── Face-based culvert flux (new path) ──
if self.culvert_face_flux_mode == "face_flux" and self._face_flux_soa is not None:
    ff = self._face_flux_soa
    native_mod.swe2d_gpu_compute_culvert_face_flux(
        n_culvert_faces=ff.structure_index.size,
        culvert_struct_idx=np.ascontiguousarray(ff.structure_index, dtype=np.int32),
        face_nx=np.ascontiguousarray(ff.face_nx, dtype=np.float64),
        face_ny=np.ascontiguousarray(ff.face_ny, dtype=np.float64),
        face_width=np.ascontiguousarray(ff.face_width, dtype=np.float64),
        donor_cell=np.ascontiguousarray(ff.donor_cell, dtype=np.int32),
        receiver_cell=np.ascontiguousarray(ff.receiver_cell, dtype=np.int32),
        invert_elev=np.ascontiguousarray(ff.invert_elev, dtype=np.float64),
        depth_safety=np.ascontiguousarray(ff.depth_safety_factor, dtype=np.float64),
        dt=dt_s,  # current time step
    )
    # Zero out culvert structure flows so source kernel skips them
    # (avoid double-counting)
    native_mod.swe2d_gpu_mask_culvert_flows(
        np.ascontiguousarray(ff.structure_index, dtype=np.int32),
    )
```

### Phase 5: Masking Culverts from Source Kernel

When `culvert_face_flux_mode == "face_flux"`, the existing `swe2d_coupling_structure_source_kernel` must not apply source/sink for culvert structures. Two options:

**Option A (Recommended): Zero-out culvert structure flows after face-flux kernel**

Add a simple kernel that sets `structure_flow[i] = 0.0` for all culvert indices:

```cuda
__global__ void swe2d_mask_culvert_source_kernel(
    int32_t n_culvert,
    const int32_t* __restrict__ culvert_indices,
    double* __restrict__ structure_flow)
{
    int32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n_culvert) return;
    structure_flow[culvert_indices[j]] = 0.0;
}
```

**Option B (Alternative): Add a `skip_culvert` mask to the source kernel** — More intrusive, requires passing an additional boolean array. Option A is simpler and sufficient.

### Phase 6: Integration with Update Kernel

**No changes required to `swe2d_update_kernel`.**

The face-based flux writes directly into `d_flux_h`, `d_flux_hu`, `d_flux_hv` — the same accumulators that `swe2d_flux_kernel` writes into. The update kernel already sums:

```cuda
for (int32_t k = s; k < e; ++k) {
    const int32_t edge = cell_edge_ids[k];
    if (edge_c0[edge] == c) {
        fh  += flux_h[edge];
        fhu += flux_hu[edge];
        fhv += flux_hv[edge];
    } else {
        fh  -= flux_h[edge];
        // ...
    }
}
```

The structure flux is written via `atomicAdd` on the **cell-level accumulators** (`flux_h[cell]`, not per-edge). This means we need a different approach.

**Revised Design**: Instead of writing into the edge-flux arrays, the face-based culvert flux should write into **per-cell net-flux accumulators** that are separate from the edge-flux arrays, but consumed by the update kernel the same way as `external_source_mps`.

Add new device arrays to `SWE2DDeviceState`:

```cuda
double* d_external_flux_h  = nullptr;   // [n_cells] net mass flux from structures
double* d_external_flux_hu = nullptr;  // [n_cells] net x-momentum flux from structures
double* d_external_flux_hv = nullptr;  // [n_cells] net y-momentum flux from structures
```

Then modify `swe2d_update_kernel` to add:

```cuda
// Add face-based structure flux
const double fh_ext  = (d_external_flux_h  ? d_external_flux_h[c]  : 0.0);
const double fhu_ext = (d_external_flux_hu ? d_external_flux_hu[c] : 0.0);
const double fhv_ext = (d_external_flux_hv ? d_external_flux_hv[c] : 0.0);

// Structure flux is already scaled by face width and has correct sign
// convention (positive = mass entering cell). It replaces source_rate for
// culverts in face_flux mode.
h_trial += dt * fh_ext * inv_a;
cell_hu[c] += dt * fhu_ext * inv_a;
cell_hv[c] += dt * fhv_ext * inv_a;
```

**But wait** — this defeats the purpose of writing into the flux arrays! The whole point was to use the same pathway.

**Final Design Decision**: Use a **separate accumulator** for structure momentum flux, because the edge-flux arrays are per-edge (CSR-indexed) not per-cell. The per-cell `external_source_mps` is only for depth (scalar). We need per-cell vectors for momentum.

The cleanest approach:

1. `d_external_source_mps` continues to handle depth source/sink (used for rain, drainage, weirs, orifices, pumps, bridges, and — in legacy mode — culverts).

2. New arrays `d_ext_struct_flux_h`, `d_ext_struct_flux_hu`, `d_ext_struct_flux_hv` hold the per-cell net flux from face-based culvert coupling. These are zeroed each step and accumulate contributions from `swe2d_culvert_face_flux_kernel`.

3. `swe2d_update_kernel` is modified to add these three terms to the update, with the same `inv_a` scaling.

### Phase 7: Testing and Validation

#### 7.1 Unit Tests

| Test | Description |
|---|---|
| `test_face_flux_soa_packing` | Verify `_build_face_flux_soa()` correctly computes normals and face widths for box/circular culverts |
| `test_face_flux_mass_conservation` | Two-cell domain with a culvert. Verify $\Delta h_u + \Delta h_d = 0$ exactly (to machine precision) |
| `test_face_flux_momentum_transfer` | Upstream cell has velocity; verify that momentum is transferred proportionally to discharge |
| `test_face_flux_depth_limiter` | Verify that the depth safety factor prevents upstream drying |
| `test_face_flux_vs_source_sink_parity` | For a simple test case, compare total mass change between face_flux and source_sink modes; they should agree within $O(\Delta x)$ |

#### 7.2 Validation Against GPU Test Suite

The existing test suite (`tests/test_swe2d_gpu_validation_perf.py`, `tests/test_swe2d_gpu_unstructured.py`) should pass with `culvert_face_flux_mode="off"` (default, unchanged behaviour). New tests should be added for `"face_flux"` mode.

#### 7.3 Conservation Audit

Run a dam-break with a culvert and compare:
- Total mass $\sum_i h_i A_i$ before and after each time step
- Total x-momentum $\sum_i h_{u,i} A_i$
- Total y-momentum $\sum_i h_{v,i} A_i$

In `face_flux` mode, mass should be conserved to round-off. In `source_sink` mode, mass is approximately conserved (within sub-step rounding).

---

## 5. File Manifest

| File | Change Type | Description |
|---|---|---|
| `swe2d/runtime/coupling.py` | Modify | Add `SWE2DCulvertFaceFluxSoA`, `_build_face_flux_soa()`, `culvert_face_flux_mode` param, GPU upload path |
| `swe2d/extensions/extension_models.py` | Modify | Add `face_width_override` and `face_flux_depth_safety` to `HydraulicStructure` metadata schema |
| `cpp/src/swe2d_gpu.cu` | Modify | Add `swe2d_culvert_face_flux_kernel`, `swe2d_mask_culvert_source_kernel`, orchestration in `swe2d_gpu_compute_coupling_full_on_device` |
| `cpp/src/swe2d_gpu.cuh` | Modify | Add `CulvertFaceFluxWorkspace`, `d_ext_struct_flux_*` arrays to `SWE2DDeviceState` |
| `cpp/src/swe2d_solver.cpp` | Modify | Add `external_struct_flux_h/hu/hv` to CPU update path (for parity testing) |
| `cpp/src/swe2d_bindings.cpp` | Modify | Add pybind11 bindings for upload and compute functions |
| `swe2d/runtime/backend.py` | Modify | Expose new GPU functions via `load_swe2d_native_module` |
| `swe2d/solver_config.py` | Modify | Add `culvert_face_flux_mode` to solver configuration |
| `swe2d_workbench_qt.py` | Modify | Add UI toggle for face_flux mode in the Structures tab |
| `tests/test_face_flux_coupling.py` | Create | New test file for face-based flux coupling |

---

## 6. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Momentum flux destabilizes wet/dry fronts | Depth limiter ($\alpha = 0.5$) and CFL check; fall back to `source_sink` mode |
| Hydrostatic pressure term creates spurious circulation | Use $h_s = \max(h_{\text{donor}} - z_s, 0)$ (depth above invert, not full WSE) |
| Double-counting if both paths are active | Explicit zeroing kernel removes culverts from source kernel |
| GPU memory increase for `d_ext_struct_flux_*` | 3 × n_cells × 8 bytes ≈ negligible for any realistic mesh |
| Performance regression from extra kernel launch | Kernel launch overhead is ~5μs; face-flux kernel is O(n_culverts) which is typically O(10–100) — negligible vs. O(n_edges) Riemann solve |
| Compatibility with CUDA graph capture | New kernel must be included in the captured graph sequence; requires graph invalidation when culvert_face_flux_mode changes |

---

## 7. Future Extensions

1. **Bridge face-flux coupling**: Bridges currently use a source/sink with empirical loss coefficients. A face-based approach with momentum transfer would be more consistent.

2. **Weir face-flux**: Weirs could also be modelled as face fluxes, with the Riemann flux capped at the weir discharge $Q_w = C_w \cdot L_w \cdot (H_{\text{up}} - z_w)^{3/2}$.

3. **Partial face occupation**: Currently, the face width $\ell_s$ is the full culvert span. Future work could model partial face occupation by integrating the Riemann flux only over the submerged fraction of the face.

4. **Multi-barrel culverts**: For `culvert_barrels > 1`, the face width should be `barrels × span` and the discharge should be divided by the number of barrels before computing per-barrel momentum flux.

---

## 8. Summary

The face-based culvert coupling mechanism replaces the cell-center source/sink approach with a proper finite-volume face flux, ensuring:

- **Exact mass conservation** by telescoping-sum property
- **Consistent momentum transfer** proportional to discharge and donor-cell velocity
- **Hydrostatic pressure** at the structure face, consistent with the SWE
- **Depth-limited discharge** to prevent upstream drying
- **Minimal code changes** — the new kernel writes into existing accumulator arrays and the update kernel needs only three extra array lookups

The feature is **opt-in** via `culvert_face_flux_mode="face_flux"` and coexists with the existing `source_sink` default. Only culvert-type structures are affected; weirs, orifices, bridges, and pumps continue to use the source/sink path.