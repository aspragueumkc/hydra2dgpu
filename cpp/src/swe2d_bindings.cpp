// swe2d_bindings.cpp
// pybind11 module: backwater_swe2d
//
// Exposes the 2D SWE hybrid GPU/CPU solver to Python as an opaque capsule-based API.
// Python users interact through swe2d_backend.py which wraps this module.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "swe2d_mesh.hpp"
#include "swe2d_solver.hpp"

#ifdef BACKWATER_HAS_CUDA
#include "swe2d_gpu.cuh"
#endif

#include <memory>
#include <stdexcept>

namespace py = pybind11;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: require a C-contiguous numpy array of given dtype
// ─────────────────────────────────────────────────────────────────────────────
template <typename T>
static const T* require_array(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& arr,
    py::ssize_t expected_size,
    const char* name)
{
    if (arr.size() != expected_size) {
        throw std::invalid_argument(
            std::string(name) + ": expected size " + std::to_string(expected_size)
            + " but got " + std::to_string(arr.size()));
    }
    return arr.data();
}

// ─────────────────────────────────────────────────────────────────────────────
// Thin Python wrapper for SWE2DMesh (holds the mesh by value)
// ─────────────────────────────────────────────────────────────────────────────
struct PyMesh {
    SWE2DMesh mesh;
};

// ─────────────────────────────────────────────────────────────────────────────
// Thin Python wrapper for SWE2DSolver (holds the solver; mesh kept alive
// via shared_ptr to PyMesh to prevent use-after-free)
// ─────────────────────────────────────────────────────────────────────────────
struct PySolver {
    std::shared_ptr<PyMesh> mesh_owner;
    SWE2DSolver*            solver = nullptr;

    ~PySolver() {
        if (solver) {
            swe2d_destroy(solver);
            solver = nullptr;
        }
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Module definition
// ─────────────────────────────────────────────────────────────────────────────
PYBIND11_MODULE(backwater_swe2d, m) {
    m.doc() = "2D SWE hybrid GPU/CPU solver on unstructured polygon mesh";

    // ── GPU query ─────────────────────────────────────────────────────────────
    m.def("swe2d_gpu_available", &swe2d_gpu_available,
          "Return True if a CUDA-capable GPU is present and the GPU path was compiled.");

#ifdef BACKWATER_HAS_CUDA
    m.def("swe2d_gpu_compute_coupling_sources",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> cell_area_m2,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> inlet_cell,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_flow_cms,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> structure_up_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> structure_down_cell,
           py::array_t<double, py::array::c_style | py::array::forcecast> structure_flow_cms)
           -> py::array_t<double>
        {
            const int32_t n_cells = static_cast<int32_t>(cell_area_m2.size());
            if (inlet_cell.size() != inlet_flow_cms.size()) {
                throw std::invalid_argument("inlet_cell and inlet_flow_cms must have the same length");
            }
            if (structure_up_cell.size() != structure_down_cell.size() ||
                structure_up_cell.size() != structure_flow_cms.size()) {
                throw std::invalid_argument(
                    "structure_up_cell, structure_down_cell, and structure_flow_cms must have the same length");
            }

            auto out = py::array_t<double>(n_cells);
            swe2d_gpu_compute_coupling_sources(
                n_cells,
                (n_cells > 0) ? cell_area_m2.data() : nullptr,
                static_cast<int32_t>(inlet_cell.size()),
                inlet_cell.size() ? inlet_cell.data() : nullptr,
                inlet_flow_cms.size() ? inlet_flow_cms.data() : nullptr,
                static_cast<int32_t>(structure_up_cell.size()),
                structure_up_cell.size() ? structure_up_cell.data() : nullptr,
                structure_down_cell.size() ? structure_down_cell.data() : nullptr,
                structure_flow_cms.size() ? structure_flow_cms.data() : nullptr,
                out.mutable_data());
            return out;
        },
        py::arg("cell_area_m2"),
        py::arg("inlet_cell"),
        py::arg("inlet_flow_cms"),
        py::arg("structure_up_cell"),
        py::arg("structure_down_cell"),
        py::arg("structure_flow_cms"),
        "Headless CUDA helper: convert inlet/structure transfer flows to per-cell depth-rate sources [m/s].");
#else
    m.def("swe2d_gpu_compute_coupling_sources",
        [](py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>) -> py::array_t<double>
        {
            throw std::runtime_error("CUDA path not compiled; swe2d_gpu_compute_coupling_sources is unavailable.");
        },
        py::arg("cell_area_m2"),
        py::arg("inlet_cell"),
        py::arg("inlet_flow_cms"),
        py::arg("structure_up_cell"),
        py::arg("structure_down_cell"),
        py::arg("structure_flow_cms"));
#endif

    // ── Mesh builder (legacy triangular triplets) ───────────────────────────
    // swe2d_build_mesh(node_x, node_y, node_z, cell_nodes,
    //                  bc_edge_node0, bc_edge_node1, bc_edge_type, bc_edge_val)
    // Returns an opaque PyMesh handle.
    m.def("swe2d_build_mesh",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> node_x,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_y,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_z,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell_nodes,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_node0,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_node1,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_type,
           py::array_t<double, py::array::c_style | py::array::forcecast> bc_val)
           -> std::shared_ptr<PyMesh>
        {
            if (node_x.size() != node_y.size() || node_x.size() != node_z.size()) {
                throw std::invalid_argument("node_x, node_y, node_z must have the same length");
            }
            int32_t n_nodes = static_cast<int32_t>(node_x.size());
            if (cell_nodes.size() % 3 != 0) {
                throw std::invalid_argument("cell_nodes length must be a multiple of 3");
            }
            int32_t n_cells = static_cast<int32_t>(cell_nodes.size() / 3);

            int32_t n_bc = static_cast<int32_t>(bc_node0.size());
            if (bc_node1.size() != static_cast<size_t>(n_bc) ||
                bc_type.size()  != static_cast<size_t>(n_bc) ||
                bc_val.size()   != static_cast<size_t>(n_bc)) {
                throw std::invalid_argument(
                    "bc_node0, bc_node1, bc_type, bc_val must all have the same length");
            }

            auto pm = std::make_shared<PyMesh>();
            pm->mesh = swe2d_build_mesh(
                node_x.data(), node_y.data(), node_z.data(), n_nodes,
                cell_nodes.data(), n_cells,
                n_bc > 0 ? bc_node0.data() : nullptr,
                n_bc > 0 ? bc_node1.data() : nullptr,
                n_bc > 0 ? bc_type.data()  : nullptr,
                n_bc > 0 ? bc_val.data()   : nullptr,
                n_bc);

            std::string err = swe2d_validate_mesh(pm->mesh);
            if (!err.empty()) {
                throw std::runtime_error("Mesh validation failed: " + err);
            }

            return pm;
        },
        py::arg("node_x"), py::arg("node_y"), py::arg("node_z"),
        py::arg("cell_nodes"),
        py::arg("bc_edge_node0"), py::arg("bc_edge_node1"),
        py::arg("bc_edge_type"),  py::arg("bc_edge_val"),
        "Build an unstructured triangular mesh from node and element arrays.\n\n"
        "Parameters\n----------\n"
        "node_x, node_y, node_z : ndarray float64, shape (N,)\n"
        "    Node coordinates and bed elevations.\n"
        "cell_nodes : ndarray int32, shape (M*3,) or (M,3)\n"
        "    Counter-clockwise node triplets per cell.\n"
        "bc_edge_node0, bc_edge_node1 : ndarray int32, shape (E,)\n"
        "    Endpoint node indices for each boundary edge specification.\n"
        "bc_edge_type : ndarray int32, shape (E,)\n"
        "    BCType value per boundary edge (0=INTERIOR,1=WALL,2=INFLOW_Q,\n"
        "    3=STAGE,4=OPEN,5=REFLECT).\n"
        "bc_edge_val : ndarray float64, shape (E,)\n"
        "    Prescribed value per boundary edge.\n"
        "Returns\n-------\n"
        "PyMesh handle (opaque).\n");

    // ── Mesh builder (polygon CSR) ──────────────────────────────────────────
    // swe2d_build_mesh_poly(node_x, node_y, node_z,
    //                      cell_face_offsets, cell_face_nodes,
    //                      bc_edge_node0, bc_edge_node1, bc_edge_type, bc_edge_val)
    m.def("swe2d_build_mesh_poly",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> node_x,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_y,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_z,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell_face_offsets,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell_face_nodes,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_node0,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_node1,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_type,
           py::array_t<double, py::array::c_style | py::array::forcecast> bc_val)
           -> std::shared_ptr<PyMesh>
        {
            if (node_x.size() != node_y.size() || node_x.size() != node_z.size()) {
                throw std::invalid_argument("node_x, node_y, node_z must have the same length");
            }
            if (cell_face_offsets.size() < 2) {
                throw std::invalid_argument("cell_face_offsets must have at least 2 entries");
            }

            int32_t n_nodes = static_cast<int32_t>(node_x.size());
            int32_t n_cells = static_cast<int32_t>(cell_face_offsets.size() - 1);
            int32_t n_face_nodes = static_cast<int32_t>(cell_face_nodes.size());

            int32_t n_bc = static_cast<int32_t>(bc_node0.size());
            if (bc_node1.size() != static_cast<size_t>(n_bc) ||
                bc_type.size()  != static_cast<size_t>(n_bc) ||
                bc_val.size()   != static_cast<size_t>(n_bc)) {
                throw std::invalid_argument(
                    "bc_node0, bc_node1, bc_type, bc_val must all have the same length");
            }

            int32_t tail = cell_face_offsets.data()[n_cells];
            if (tail != n_face_nodes) {
                throw std::invalid_argument(
                    "cell_face_offsets tail must equal len(cell_face_nodes)");
            }

            auto pm = std::make_shared<PyMesh>();
            pm->mesh = swe2d_build_mesh_poly(
                node_x.data(), node_y.data(), node_z.data(), n_nodes,
                cell_face_offsets.data(), cell_face_nodes.data(), n_cells,
                n_bc > 0 ? bc_node0.data() : nullptr,
                n_bc > 0 ? bc_node1.data() : nullptr,
                n_bc > 0 ? bc_type.data()  : nullptr,
                n_bc > 0 ? bc_val.data()   : nullptr,
                n_bc);

            std::string err = swe2d_validate_mesh(pm->mesh);
            if (!err.empty()) {
                throw std::runtime_error("Mesh validation failed: " + err);
            }

            return pm;
        },
        py::arg("node_x"), py::arg("node_y"), py::arg("node_z"),
        py::arg("cell_face_offsets"), py::arg("cell_face_nodes"),
        py::arg("bc_edge_node0"), py::arg("bc_edge_node1"),
        py::arg("bc_edge_type"), py::arg("bc_edge_val"),
        "Build an unstructured polygon mesh from node and CSR cell topology arrays.\n\n"
        "Parameters\n----------\n"
        "cell_face_offsets : ndarray int32, shape (M+1,)\n"
        "    CSR offsets into cell_face_nodes per cell.\n"
        "cell_face_nodes : ndarray int32, shape (K,)\n"
        "    Concatenated node rings for all polygon cells (CCW preferred).\n"
        "Returns\n-------\n"
        "PyMesh handle (opaque).\n");

    // ── Mesh info ─────────────────────────────────────────────────────────────
    m.def("swe2d_mesh_info",
        [](const std::shared_ptr<PyMesh>& pm) -> py::dict {
            if (!pm) throw std::invalid_argument("null mesh handle");
            py::dict d;
            d["n_nodes"] = pm->mesh.n_nodes;
            d["n_cells"] = pm->mesh.n_cells;
            d["n_edges"] = pm->mesh.n_edges;
            return d;
        },
        py::arg("mesh"),
        "Return dict with n_nodes, n_cells, n_edges.");

    // ── Boundary edges + runtime BC updates ─────────────────────────────────
    m.def("swe2d_boundary_edges",
        [](const std::shared_ptr<PyMesh>& pm)
            -> std::tuple<py::array_t<int32_t>, py::array_t<int32_t>, py::array_t<int32_t>, py::array_t<int32_t>, py::array_t<double>>
        {
            if (!pm) throw std::invalid_argument("null mesh handle");

            std::vector<int32_t> edge_idx;
            std::vector<int32_t> n0;
            std::vector<int32_t> n1;
            std::vector<int32_t> bc_type;
            std::vector<double> bc_val;

            edge_idx.reserve(static_cast<size_t>(pm->mesh.n_edges));
            n0.reserve(static_cast<size_t>(pm->mesh.n_edges));
            n1.reserve(static_cast<size_t>(pm->mesh.n_edges));
            bc_type.reserve(static_cast<size_t>(pm->mesh.n_edges));
            bc_val.reserve(static_cast<size_t>(pm->mesh.n_edges));

            for (int32_t e = 0; e < pm->mesh.n_edges; ++e) {
                if (pm->mesh.edge_c1[e] != -1) continue;
                edge_idx.push_back(e);
                n0.push_back(pm->mesh.edge_n0[e]);
                n1.push_back(pm->mesh.edge_n1[e]);
                bc_type.push_back(static_cast<int32_t>(pm->mesh.edge_bc[e]));
                bc_val.push_back(pm->mesh.edge_bc_val[e]);
            }

            py::array_t<int32_t> edge_idx_arr(edge_idx.size());
            py::array_t<int32_t> n0_arr(n0.size());
            py::array_t<int32_t> n1_arr(n1.size());
            py::array_t<int32_t> bc_type_arr(bc_type.size());
            py::array_t<double> bc_val_arr(bc_val.size());
            std::copy(edge_idx.begin(), edge_idx.end(), edge_idx_arr.mutable_data());
            std::copy(n0.begin(), n0.end(), n0_arr.mutable_data());
            std::copy(n1.begin(), n1.end(), n1_arr.mutable_data());
            std::copy(bc_type.begin(), bc_type.end(), bc_type_arr.mutable_data());
            std::copy(bc_val.begin(), bc_val.end(), bc_val_arr.mutable_data());

            return {edge_idx_arr, n0_arr, n1_arr, bc_type_arr, bc_val_arr};
        },
        py::arg("mesh"),
        "Return boundary edge arrays: (edge_index, node0, node1, bc_type, bc_val).");

    m.def("swe2d_set_boundary_values",
        [](const std::shared_ptr<PyMesh>& pm,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> edge_index,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_type,
           py::array_t<double, py::array::c_style | py::array::forcecast> bc_val) {
            if (!pm) throw std::invalid_argument("null mesh handle");
            if (edge_index.size() != bc_type.size() || edge_index.size() != bc_val.size()) {
                throw std::invalid_argument("edge_index, bc_type, bc_val must have same length");
            }
            for (py::ssize_t i = 0; i < edge_index.size(); ++i) {
                int32_t e = edge_index.data()[i];
                if (e < 0 || e >= pm->mesh.n_edges) {
                    throw std::invalid_argument("edge_index out of range");
                }
                if (pm->mesh.edge_c1[e] != -1) {
                    throw std::invalid_argument("edge_index refers to interior edge");
                }
                pm->mesh.edge_bc[e] = static_cast<BCType>(bc_type.data()[i]);
                pm->mesh.edge_bc_val[e] = bc_val.data()[i];
            }
        },
        py::arg("mesh"), py::arg("edge_index"), py::arg("bc_type"), py::arg("bc_val"),
        "Update boundary condition type/value for boundary edges by edge index.");

    m.def("swe2d_solver_set_boundary_values",
        [](const std::shared_ptr<PySolver>& ps,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> edge_index,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_type,
           py::array_t<double, py::array::c_style | py::array::forcecast> bc_val) {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            if (edge_index.size() != bc_type.size() || edge_index.size() != bc_val.size()) {
                throw std::invalid_argument("edge_index, bc_type, bc_val must have same length");
            }
            swe2d_solver_set_boundary_values(ps->solver,
                                             edge_index.data(),
                                             bc_type.data(),
                                             bc_val.data(),
                                             static_cast<int32_t>(edge_index.size()));
        },
        py::arg("solver"), py::arg("edge_index"), py::arg("bc_type"), py::arg("bc_val"),
        "Update boundary condition values on an active solver and sync GPU arrays.");

    m.def("swe2d_solver_set_boundary_hydrographs",
        [](const std::shared_ptr<PySolver>& ps,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> edge_index,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> bc_type,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> offsets,
           py::array_t<double, py::array::c_style | py::array::forcecast> time_s,
           py::array_t<double, py::array::c_style | py::array::forcecast> value) {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            if (edge_index.size() != bc_type.size()) {
                throw std::invalid_argument("edge_index and bc_type must have same length");
            }
            if (offsets.size() != edge_index.size() + 1) {
                throw std::invalid_argument("offsets length must be n_edges + 1");
            }
            swe2d_solver_set_boundary_hydrographs(ps->solver,
                                                  edge_index.data(),
                                                  bc_type.data(),
                                                  offsets.data(),
                                                  time_s.data(),
                                                  value.data(),
                                                  static_cast<int32_t>(edge_index.size()),
                                                  static_cast<int32_t>(time_s.size()));
        },
        py::arg("solver"), py::arg("edge_index"), py::arg("bc_type"), py::arg("offsets"), py::arg("time_s"), py::arg("value"),
        "Register per-boundary-edge hydrograph timeseries on the solver.");

    m.def("swe2d_solver_set_rain_cn_forcing",
        [](const std::shared_ptr<PySolver>& ps,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell_gage_idx,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> gage_offsets,
           py::array_t<double, py::array::c_style | py::array::forcecast> hg_time_s,
           py::array_t<double, py::array::c_style | py::array::forcecast> hg_cum_mm,
           py::array_t<double, py::array::c_style | py::array::forcecast> cn,
           double ia_ratio,
           double mm_to_model_depth) {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            if (cell_gage_idx.size() != cn.size()) {
                throw std::invalid_argument("cell_gage_idx and cn must have same length");
            }
            if (gage_offsets.size() < 2) {
                throw std::invalid_argument("gage_offsets must contain at least 2 entries");
            }
            if (hg_time_s.size() != hg_cum_mm.size()) {
                throw std::invalid_argument("hg_time_s and hg_cum_mm must have same length");
            }
            swe2d_solver_set_rain_cn_forcing(ps->solver,
                                             cell_gage_idx.data(),
                                             gage_offsets.data(),
                                             hg_time_s.data(),
                                             hg_cum_mm.data(),
                                             cn.data(),
                                             static_cast<int32_t>(cell_gage_idx.size()),
                                             static_cast<int32_t>(gage_offsets.size() - 1),
                                             static_cast<int32_t>(hg_time_s.size()),
                                             ia_ratio,
                                             mm_to_model_depth);
        },
        py::arg("solver"), py::arg("cell_gage_idx"), py::arg("gage_offsets"), py::arg("hg_time_s"), py::arg("hg_cum_mm"), py::arg("cn"), py::arg("ia_ratio") = 0.2, py::arg("mm_to_model_depth") = 1.0e-3,
        "Register per-cell rain/CN forcing data on the solver.");

    // ── Solver creation ───────────────────────────────────────────────────────
    m.def("swe2d_create_solver",
        [](std::shared_ptr<PyMesh> pm,
           py::array_t<double, py::array::c_style | py::array::forcecast> h0,
           py::object hu0_obj,
           py::object hv0_obj,
           py::object n_mann_cell_obj,
           double g, double n_mann, double h_min,
           double cfl, double dt_max, double dt_fixed,
                  double max_inv_area,
                  double cfl_lambda_cap,
                  double momentum_cap_min_speed,
                  double momentum_cap_celerity_mult,
                  double depth_cap,
                  double max_rel_depth_increase,
                  double shallow_damping_depth,
                  int gpu_diag_sync_interval_steps,
              bool use_gpu, int n_threads,
              int temporal_order,
              int spatial_scheme,
              int turbulence_model,
              int bed_friction_model,
              bool enable_rain_module,
              bool enable_pipe_network_module,
              bool enable_hydraulic_structures,
              int degen_mode,
              double front_flux_damping,
              bool   active_set_hysteresis)
           -> std::shared_ptr<PySolver>
        {
            if (!pm) throw std::invalid_argument("null mesh handle");
            int32_t nc = pm->mesh.n_cells;

            if (h0.size() != static_cast<size_t>(nc)) {
                throw std::invalid_argument("h0 length must equal n_cells");
            }

            const double* hu0_ptr = nullptr;
            const double* hv0_ptr = nullptr;
            const double* n_mann_cell_ptr = nullptr;
            py::array_t<double, py::array::c_style | py::array::forcecast> hu0_arr, hv0_arr;
            py::array_t<double, py::array::c_style | py::array::forcecast> n_mann_cell_arr;

            if (!hu0_obj.is_none()) {
                hu0_arr = hu0_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                if (hu0_arr.size() != static_cast<size_t>(nc))
                    throw std::invalid_argument("hu0 length must equal n_cells");
                hu0_ptr = hu0_arr.data();
            }
            if (!hv0_obj.is_none()) {
                hv0_arr = hv0_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                if (hv0_arr.size() != static_cast<size_t>(nc))
                    throw std::invalid_argument("hv0 length must equal n_cells");
                hv0_ptr = hv0_arr.data();
            }
            if (!n_mann_cell_obj.is_none()) {
                n_mann_cell_arr = n_mann_cell_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                if (n_mann_cell_arr.size() != static_cast<size_t>(nc))
                    throw std::invalid_argument("n_mann_cell length must equal n_cells");
                n_mann_cell_ptr = n_mann_cell_arr.data();
            }

            SWE2DSolverConfig cfg;
            cfg.g         = g;
            cfg.n_mann    = n_mann;
            cfg.h_min     = h_min;
            cfg.cfl       = cfl;
            cfg.dt_max    = dt_max;
            cfg.dt_fixed  = dt_fixed;
            cfg.max_inv_area = max_inv_area;
            cfg.cfl_lambda_cap = cfl_lambda_cap;
            cfg.momentum_cap_min_speed = momentum_cap_min_speed;
            cfg.momentum_cap_celerity_mult = momentum_cap_celerity_mult;
            cfg.depth_cap = depth_cap;
            cfg.max_rel_depth_increase = max_rel_depth_increase;
            cfg.shallow_damping_depth = shallow_damping_depth;
            cfg.gpu_diag_sync_interval_steps = gpu_diag_sync_interval_steps;
            cfg.temporal_order = temporal_order;
            cfg.spatial_scheme = spatial_scheme;
            cfg.turbulence_model = turbulence_model;
            cfg.bed_friction_model = bed_friction_model;
            cfg.enable_rain_module = enable_rain_module;
            cfg.enable_pipe_network_module = enable_pipe_network_module;
            cfg.enable_hydraulic_structures = enable_hydraulic_structures;
            cfg.use_gpu   = use_gpu;
            cfg.n_threads = n_threads;
            cfg.degen_mode = degen_mode;
            cfg.front_flux_damping = front_flux_damping;
            cfg.active_set_hysteresis = active_set_hysteresis;

            auto ps = std::make_shared<PySolver>();
            ps->mesh_owner = pm;
            ps->solver = swe2d_create(pm->mesh, h0.data(), hu0_ptr, hv0_ptr, n_mann_cell_ptr, cfg);
            return ps;
        },
        py::arg("mesh"),
        py::arg("h0"),
        py::arg("hu0")      = py::none(),
        py::arg("hv0")      = py::none(),
        py::arg("n_mann_cell") = py::none(),
        py::arg("g")        = 9.81,
        py::arg("n_mann")   = 0.035,
        py::arg("h_min")    = 1.0e-6,
        py::arg("cfl")      = 0.45,
        py::arg("dt_max")   = 10.0,
        py::arg("dt_fixed") = -1.0,
        py::arg("max_inv_area") = 1.0e6,
        py::arg("cfl_lambda_cap") = 1.0e6,
        py::arg("momentum_cap_min_speed") = 50.0,
        py::arg("momentum_cap_celerity_mult") = 20.0,
        py::arg("depth_cap") = 1.0e6,
        py::arg("max_rel_depth_increase") = 2.0,
        py::arg("shallow_damping_depth") = 1.0e-4,
        py::arg("gpu_diag_sync_interval_steps") = 1,
        py::arg("use_gpu")  = true,
        py::arg("n_threads") = 0,
        py::arg("temporal_order") = 2,
        py::arg("spatial_scheme") = 0,
        py::arg("turbulence_model") = 0,
        py::arg("bed_friction_model") = 0,
        py::arg("enable_rain_module") = false,
        py::arg("enable_pipe_network_module") = false,
        py::arg("enable_hydraulic_structures") = false,
        py::arg("degen_mode") = 0,
        py::arg("front_flux_damping") = 0.5,
        py::arg("active_set_hysteresis") = true,
        "Create a 2D SWE solver.\n\n"
        "Parameters\n----------\n"
        "mesh : PyMesh handle from swe2d_build_mesh\n"
        "h0   : ndarray float64, shape (M,) — initial water depth\n"
        "hu0  : ndarray float64, shape (M,) or None — initial x-momentum\n"
        "hv0  : ndarray float64, shape (M,) or None — initial y-momentum\n"
        "Returns PySolver handle.\n");

    // ── Step ──────────────────────────────────────────────────────────────────
    m.def("swe2d_step",
        [](std::shared_ptr<PySolver>& ps, double dt_request) -> py::dict
        {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            SWE2DStepDiag diag = swe2d_step(ps->solver, dt_request);
            py::dict d;
            d["dt"]         = diag.dt;
            d["wet_cells"]  = diag.wet_cells;
            d["max_depth"]  = diag.max_depth;
            d["min_depth"]  = diag.min_depth;
            d["mass_total"] = diag.mass_total;
            d["max_courant"] = diag.max_courant;
            d["max_depth_residual"] = diag.max_depth_residual;
            d["max_wse_elev_error"] = diag.max_wse_elev_error;
            d["gpu_active"] = diag.gpu_active;
            return d;
        },
        py::arg("solver"), py::arg("dt_request") = -1.0,
        "Advance one timestep.  Returns diagnostics dict.");

    // ── Get state ─────────────────────────────────────────────────────────────
    m.def("swe2d_get_state",
        [](const std::shared_ptr<PySolver>& ps)
            -> std::tuple<py::array_t<double>, py::array_t<double>, py::array_t<double>>
        {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            int32_t nc = ps->solver->mesh->n_cells;

            auto h_out  = py::array_t<double>(nc);
            auto hu_out = py::array_t<double>(nc);
            auto hv_out = py::array_t<double>(nc);

            // swe2d_get_state routes directly device→caller when GPU is active;
            // no host mirror update — state stays device-resident.
            swe2d_get_state(ps->solver,
                h_out.mutable_data(), hu_out.mutable_data(), hv_out.mutable_data());
            return {h_out, hu_out, hv_out};
        },
        py::arg("solver"),
        "Return current (h, hu, hv) state arrays.");

    // ── Set state ─────────────────────────────────────────────────────────────
    m.def("swe2d_set_state",
        [](std::shared_ptr<PySolver>& ps,
           py::array_t<double, py::array::c_style | py::array::forcecast> h_in,
           py::array_t<double, py::array::c_style | py::array::forcecast> hu_in,
           py::array_t<double, py::array::c_style | py::array::forcecast> hv_in)
        {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            const int32_t nc = ps->solver->mesh->n_cells;
            require_array(h_in, nc, "h_in");
            require_array(hu_in, nc, "hu_in");
            require_array(hv_in, nc, "hv_in");
            swe2d_set_state(ps->solver, h_in.data(), hu_in.data(), hv_in.data());
        },
        py::arg("solver"), py::arg("h_in"), py::arg("hu_in"), py::arg("hv_in"),
        "Overwrite current (h, hu, hv) solver state arrays.");

    // ── Destroy ───────────────────────────────────────────────────────────────
    m.def("swe2d_destroy",
        [](std::shared_ptr<PySolver>& ps) {
            if (ps && ps->solver) {
                swe2d_destroy(ps->solver);
                ps->solver = nullptr;
            }
        },
        py::arg("solver"),
        "Explicitly free native solver resources (also called on GC).");

    // ── PyMesh / PySolver as opaque Python types ──────────────────────────────
    py::class_<PyMesh, std::shared_ptr<PyMesh>>(m, "SWE2DMeshHandle")
        .def("__repr__", [](const PyMesh& pm) {
            return "<SWE2DMeshHandle nodes=" + std::to_string(pm.mesh.n_nodes)
                 + " cells=" + std::to_string(pm.mesh.n_cells)
                 + " edges=" + std::to_string(pm.mesh.n_edges) + ">";
        });

    py::class_<PySolver, std::shared_ptr<PySolver>>(m, "SWE2DSolverHandle")
        .def("__repr__", [](const PySolver& ps) {
            return std::string("<SWE2DSolverHandle ") +
                   (ps.solver ? ("t=" + std::to_string(ps.solver->t)) : "destroyed") + ">";
        });

    // ── BCType constants ──────────────────────────────────────────────────────
    py::class_<BCType>(m, "BCType");
    m.attr("BC_INTERIOR") = py::int_(static_cast<int>(BCType::INTERIOR));
    m.attr("BC_WALL")     = py::int_(static_cast<int>(BCType::WALL));
    m.attr("BC_INFLOW_Q") = py::int_(static_cast<int>(BCType::INFLOW_Q));
    m.attr("BC_STAGE")    = py::int_(static_cast<int>(BCType::STAGE));
    m.attr("BC_OPEN")     = py::int_(static_cast<int>(BCType::OPEN));
    m.attr("BC_REFLECT")  = py::int_(static_cast<int>(BCType::REFLECT));
    m.attr("BC_NORMAL_DEPTH") = py::int_(static_cast<int>(BCType::NORMAL_DEPTH));
}
