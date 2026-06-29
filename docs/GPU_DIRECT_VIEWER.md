# GPU-Direct Real-Time Viewer

## Goal

Render simulation state (h, hu, hv, speed) to a display window with **zero D2H transfers** during the run. Drive the display directly from GPU memory, the way games render frames. No NumPy readback, no Python rasterization, no PCIe bottleneck.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    CUDA Stream                           │
│  ┌──────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ Solver   │  │ Color compute    │  │ Write to      │  │
│  │ Step     │──▶ kernel (per cell)│──▶ VBO (shared   │  │
│  │ (d_h)    │  │ h → RGBA        │  │ with OpenGL)  │  │
│  └──────────┘  └──────────────────┘  └──────┬────────┘  │
│                                              │           │
└──────────────────────────────────────────────┼───────────┘
                                               │ no D2H
          ┌────────────────────────────────────┘
          ▼
┌──────────────────┐     ┌──────────────────┐
│  OpenGL context   │────▶│  QOpenGLWidget   │───▶ Display
│  (QGIS shared or  │     │  (separate       │
│   own context)    │     │   dialog window) │
└──────────────────┘     └──────────────────┘
```

## Components

### 1. CUDA Color Kernel (`swe2d_gpu_viewer.cu`)

```cuda
// One thread per cell.  Reads d_h (or d_hu/d_hv/speed),
// maps value → RGBA via a colormap lookup table, writes to
// a vertex attribute buffer shared with OpenGL.

__global__ void viewer_color_kernel(
    int32_t n_cells,
    const double* __restrict__ d_field,   // h, hu, hv, or speed
    double field_min,
    double field_max,
    const float* __restrict__ d_colormap, // 256×4 RGBA LUT on device
    float* __restrict__ d_vbo_colors      // [n_cells * 3 * 4] floats (3 verts per tri × RGBA)
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n_cells) return;
    double v = (d_field[c] - field_min) / fmax(field_max - field_min, 1e-12);
    v = fmax(0.0, fmin(1.0, v));
    int idx = min(255, max(0, int(v * 255.0)));
    float4 color = ((const float4*)d_colormap)[idx];
    // Write to 3 triangle vertices (same color for all)
    for (int k = 0; k < 3; k++) {
        ((float4*)d_vbo_colors)[c * 3 + k] = color;
    }
}
```

### 2. CUDA-OpenGL Interop

The vertex buffer is registered for interop so both CUDA and OpenGL access the same device memory:

```cpp
// Allocation (once at init):
GLuint vbo;
glGenBuffers(1, &vbo);
glBindBuffer(GL_ARRAY_BUFFER, vbo);
glBufferData(GL_ARRAY_BUFFER, n_cells * 3 * 4 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

cudaGraphicsResource* cuda_vbo;
cudaGraphicsGLRegisterBuffer(&cuda_vbo, vbo, cudaGraphicsMapFlagsWriteDiscard);

// Each frame (after solver step):
cudaGraphicsMapResources(1, &cuda_vbo, stream);
float* d_vbo_colors;
size_t num_bytes;
cudaGraphicsResourceGetMappedPointer((void**)&d_vbo_colors, &num_bytes, cuda_vbo);

viewer_color_kernel<<<grid, block, 0, stream>>>(n_cells, d_h, ... d_vbo_colors);

cudaGraphicsUnmapResources(1, &cuda_vbo, stream);

// OpenGL renders VBO directly — no D2H involved.
```

### 3. QOpenGLWidget Dialog (`swe2d/workbench/views/gpu_viewer_dialog.py`)

A standalone floating window (QDialog + QOpenGLWidget):

```python
class GPUViewerDialog(QDialog):
    """Floating window that renders simulation state directly from GPU."""

    def __init__(self, parent=None, backend=None, mesh_data=None):
        super().__init__(parent)
        self.setWindowTitle("GPU Direct Viewer")
        self.setGeometry(100, 100, 1280, 720)
        layout = QVBoxLayout(self)
        self.gl_widget = GPUViewerGLWidget(backend, mesh_data)
        layout.addWidget(self.gl_widget)

        # Controls toolbar
        controls = QHBoxLayout()
        self.field_combo = QComboBox()
        self.field_combo.addItems(["Depth (h)", "Speed", "Momentum X", "Momentum Y"])
        self.auto_scale_cb = QCheckBox("Auto-scale")
        self.auto_scale_cb.setChecked(True)
        controls.addWidget(QLabel("Field:"))
        controls.addWidget(self.field_combo)
        controls.addWidget(self.auto_scale_cb)
        layout.addLayout(controls)

        # Timer pulls latest state from device at refresh rate
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(16)  # ~60 FPS

    def _refresh(self):
        self.gl_widget.update()  # triggers paintGL — CUDA color kernel runs there
```

### 4. OpenGL Shaders

**Vertex shader** — transforms mesh vertices from model coords to screen:

```glsl
#version 330
layout(location = 0) in vec2 a_pos;    // node_x, node_y
layout(location = 1) in vec4 a_color;  // RGBA from CUDA kernel
out vec4 v_color;
uniform mat4 u_proj;  // orthographic projection from CRS coords to screen
void main() {
    gl_Position = u_proj * vec4(a_pos, 0.0, 1.0);
    v_color = a_color;
}
```

**Fragment shader:**

```glsl
#version 330
in vec4 v_color;
out vec4 fragColor;
void main() {
    fragColor = v_color;
}
```

### 5. Basemap Support

Two options:

**A. QGIS map image as texture (easy):** Before the CUDA color kernel runs, render the QGIS map canvas at the current extent to a QImage, upload as OpenGL texture, and draw it as a full-screen quad behind the mesh. This gives you full QGIS basemap support (satellite, streets, etc.).

```python
qimage = self.iface.mapCanvas().saveAsImage(viewport_size)
# Upload to OpenGL texture
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, qimage.width(), qimage.height(),
             0, GL_RGBA, GL_UNSIGNED_BYTE, qimage.bits())
```

**B. Slippy-map tiles (hard):** Fetch and cache XYZ tiles directly, bypassing QGIS. More work but fully independent of the QGIS canvas.

### 6. Initial Limitations

| Feature | Status |
|---------|--------|
| Real-time GPU-direct coloring | Needs new CUDA kernel + interop |
| Separate QDialog window | New widget class |
| Basemap texture from QGIS | ~10 lines of Python |
| Field selector (h/hu/hv/speed) | Combo box → kernel param |
| Auto-scale per frame | Scan min/max via thrust::reduce |
| Screenshot via external tool | Works OOTB (window is a regular OS window) |
| Recording to video | External tool (OBS, etc.) |
| Timeline scrub | Not needed in live mode — always shows latest step |
| CLUDA-OpenGL interop | QGIS must expose its GL context or viewer uses its own |

### 7. Implementation Priority

1. **Phase 1** — `viewer_color_kernel` in C++ + `cudaGraphicsGLRegisterBuffer` binding
2. **Phase 2** — `GPUViewerGLWidget` with minimal OpenGL renderer (colored triangles, no basemap)
3. **Phase 3** — `GPUViewerDialog` with controls (field selector, auto-scale, start/stop)
4. **Phase 4** — Basemap texture from QGIS canvas
5. **Phase 5** — Integration with run controller (auto-launch on run start, optional)

### 8. GPU-Direct Diagnostics & NVENC Recording

**The problem:** At dt=0.05s for a 3-hour simulation (216,000 steps), every step triggers a `cudaMemcpyDeviceToHost` of the 3-value `d_diag_packed` array (lines 5486, 6124, 6318, 6517 of `swe2d_gpu.cu`). Each transfer is tiny in bytes but still forces a pipeline drain. Cumulatively this kills GPU utilization.

**The fix:** Same ring buffer pattern as the snapshots — but for diagnostics. The diagnostics are already computed on-device by `pack_diag_kernel`. Instead of reading them back to host every step, push them to a device ring buffer. Then render them as pixels into the viewer framebuffer.

#### 8a. Diagnostic Ring Buffer (Device)

Like the snapshot ring buffer but storing per-step diagnostic records instead of full state arrays.

```cuda
// Device-side diagnostic record (stored per step)
struct DiagRecord {
    double max_courant;
    double max_wse_error;
    double wet_cells;
    double t_s;          // simulation time
    double dt_used;       // timestep taken
    double gpu_active;    // 1 if GPU was active
    double mass_total;    // total mass in system
};

// Device ring buffer (added to SWE2DDeviceState)
DiagRecord* d_diag_ring = nullptr;  // [diag_ring_capacity]
int32_t diag_ring_capacity = 0;
int32_t diag_ring_count = 0;
```

Each step, after `pack_diag_kernel` writes to `d_diag_packed`, a push kernel copies the 3 packed values + additional state into the next ring slot:

```cuda
__global__ void push_diag_kernel(
    const double* __restrict__ d_diag_packed,  // [3]: max_lambda, max_wse_err, wet_cells
    double t_s,
    double dt_used,
    int32_t slot,
    DiagRecord* __restrict__ d_diag_ring)
{
    d_diag_ring[slot].max_courant  = d_diag_packed[0];
    d_diag_ring[slot].max_wse_error = d_diag_packed[1];
    d_diag_ring[slot].wet_cells    = d_diag_packed[2];
    d_diag_ring[slot].t_s          = t_s;
    d_diag_ring[slot].dt_used      = dt_used;
}
```

No D2H transfer — pure device-local write.

#### 8b. HUD Render Kernel

A CUDA kernel that renders diagnostic values as text/graphics into the framebuffer. Uses a **font glyph atlas** — a device texture containing pre-rendered characters:

```cuda
__global__ void hud_render_kernel(
    const DiagRecord* __restrict__ d_diag_ring,
    int32_t diag_count,
    const uint8_t* __restrict__ d_font_atlas,  // 256 × glyph_width × glyph_height
    uint8_t* __restrict__ d_framebuffer,       // RGBA, width × height
    int fb_width,
    int fb_height)
{
    // Read latest diagnostic record
    DiagRecord r = d_diag_ring[diag_count - 1];

    // Render each value as text at a fixed position
    // "t= 3600.0s  dt=0.050  CFL=0.42  wet=18423/32000  mass=1.24e6"
    //
    // Each glyph is rendered by copying from the font atlas to the
    // framebuffer at the correct screen position.
    //
    // A simple approach: render a semi-transparent dark bar at the
    // bottom of the screen, then draw white text on top.
}
```

The font atlas is a 16×16 grid of 8×16 pixel glyphs (ASCII 32-127), pre-rendered on the CPU and uploaded as a device texture once at init. The HUD kernel samples this atlas to draw each character into the framebuffer.

#### 8c. NVENC Hardware Encoding Pipeline

The complete zero-D2H pipeline:

```
Solver step
  → push_diag_kernel writes to d_diag_ring (device, async)
  → (optionally) viewer_color_kernel writes to VBO → OpenGL draw
  → (after N solver steps or at 60Hz) HUD render writes to framebuffer
  → framebuffer (CUDA array) → NVENC → h.264/h.265 bitstream → .mp4/.ts file
```

NVENC integration:

```cpp
#include <nvEncodeAPI.h>

// 1. Create NVENC encoder
NV_ENCODE_API_FUNCTION_LIST enc;
NvEncodeAPICreateInstance(&enc);

// 2. Configure encoding session
NV_ENC_INITIALIZE_PARAMS init_params = {};
init_params.encodeGUID = NV_ENC_CODEC_H264_GUID;
init_params.encodeWidth = fb_width;
init_params.encodeHeight = fb_height;
init_params.enableEncodeAsync = true;
enc.nvEncInitializeEncoder(encoder, &init_params);

// 3. Every frame: submit CUDA array to NVENC input surface
NV_ENC_PIC_PARAMS pic_params = {};
// Map CUDA array → NVENC input surface (no D2H, device-to-encoder on-GPU)
enc.nvEncEncodePicture(encoder, &pic_params);

// 4. Get encoded bitstream
NV_ENC_LOCK_BITSTREAM lock = {};
enc.nvEncLockBitstream(encoder, &lock);
// write lock.bitstreamBuffer to .mp4 file via libavformat
enc.nvEncUnlockBitstream(encoder, &lock);
```

NVENC is a fixed-function hardware block on the GPU. It reads from dedicated video memory — the CUDA framebuffer array is registered as an NVENC input surface via `NvEncRegisterResource` with `NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPOINTER`. The encoder DMA's the frame directly from GPU memory — **zero PCIe transfers, zero D2H**.

#### 8d. What This Replaces

| Current | GPU-Direct Replacement |
|---------|----------------------|
| `cudaMemcpy(d_diag_packed → host)` every step | `push_diag_kernel` → device ring buffer |
| Python `last_diag` dict processing | HUD render kernel → framebuffer pixels |
| Text log file (`log.txt`) | Video file with visual HUD (`.mp4`) |
| `perf_mode` toggle for skipping D2H | Irrelevant — no D2H to skip |
| Progress bar text update | Visual progress bar in HUD |

#### 8e. Work Breakdown

| Component | Effort | Description |
|-----------|--------|-------------|
| Diagnostic ring buffer struct + arrays | Small | Add `DiagRecord` struct, `d_diag_ring`, push kernel, grow/cleanup |
| `push_diag_kernel` | Small | 1-block kernel, called at end of every `swe2d_gpu_step` |
| Font glyph atlas | Medium | Generate or load a bitmap font texture, upload to device |
| HUD render kernel | Medium | CUDA 2D kernel rendering text + graphs into framebuffer |
| NVENC integration | Large | Link `nvEncodeAPI`, encoder session lifecycle, CUDA array → NVENC |
| Recording UI controls | Medium | Start/stop recording, filename, quality presets |
| Playback of recorded diagnostics | Medium | Decode video, parse HUD frames (same as watching any video) |

#### 8f. Recording Modes

**Live mode (interactive):** Viewer window + optional NVENC recording to file. User sees the simulation in real-time with HUD overlay. Recording can be started/stopped mid-run.

**Headless record mode (CLI):** No window, no display. Frames go directly from CUDA color kernel → HUD render → NVENC → file. Same pipeline, just skipping the OpenGL present step.

```
hydra_viewer --mode record --output sim_run_001.mp4 --fps 30 --quality high
```

This makes the "log" for a 3-hour simulation a playable video file. At 30 FPS, that's 324,000 frames → ~10GB for high-quality h.264 (vs gigabytes of text logs that are unreadable). You can pause, scrub, and visually inspect any timestep.

### 9. Updated Implementation Priority

1. **Phase 1** — `viewer_color_kernel` in C++ + `cudaGraphicsGLRegisterBuffer` binding
2. **Phase 2** — `GPUViewerGLWidget` with minimal OpenGL renderer (colored triangles, no basemap)
3. **Phase 3** — `GPUViewerDialog` with controls (field selector, auto-scale, start/stop)
4. **Phase 4** — Diagnostic ring buffer + `push_diag_kernel` (remove per-step D2H readback)
5. **Phase 5** — Font atlas + HUD render kernel (show diagnostics on viewer)
6. **Phase 6** — NVENC integration (record viewer output to .mp4)
7. **Phase 7** — Basemap texture from QGIS canvas
8. **Phase 8** — Headless CLI recording mode
9. **Phase 9** — Integration with run controller (auto-launch on run start, recording controls)
