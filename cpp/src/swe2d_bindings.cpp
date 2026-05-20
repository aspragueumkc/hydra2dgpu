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

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

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

    m.def("swe2d_gpu_drainage_step",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> cell_wse,
           py::array_t<double, py::array::c_style | py::array::forcecast> cell_area,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_invert_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_max_depth,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_surface_area,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> link_from,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> link_to,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_length,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_roughness_n,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_diameter,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_max_flow,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> inlet_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> inlet_node,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_crest_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_width,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_coefficient,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_max_capture,
              py::array_t<int32_t, py::array::c_style | py::array::forcecast> outfall_cell,
              py::array_t<int32_t, py::array::c_style | py::array::forcecast> outfall_node,
              py::array_t<double, py::array::c_style | py::array::forcecast> outfall_invert_elev,
              py::array_t<double, py::array::c_style | py::array::forcecast> outfall_diameter,
              py::array_t<double, py::array::c_style | py::array::forcecast> outfall_coefficient,
              py::array_t<double, py::array::c_style | py::array::forcecast> outfall_max_flow,
              py::array_t<int32_t, py::array::c_style | py::array::forcecast> outfall_zero_storage,
                  py::array_t<int32_t, py::array::c_style | py::array::forcecast> pipe_end_cell,
                  py::array_t<int32_t, py::array::c_style | py::array::forcecast> pipe_end_node,
                  py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_invert_elev,
                  py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_diameter,
                  py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_area,
                  py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_inlet_loss_k,
                  py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_outlet_loss_k,
              py::array_t<double, py::array::c_style | py::array::forcecast> cell_depth,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_depth,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_flow,
           double dt_s,
           double gravity,
              int32_t solver_mode,
              double head_deadband_m,
              double dynamic_flow_relaxation)
           -> py::tuple
        {
            const int32_t n_cells = static_cast<int32_t>(cell_wse.size());
            if (cell_area.size() != static_cast<size_t>(n_cells)) {
                throw std::invalid_argument("cell_wse and cell_area must have same length");
            }
            const int32_t n_nodes = static_cast<int32_t>(node_invert_elev.size());
            if (node_max_depth.size() != static_cast<size_t>(n_nodes) ||
                node_surface_area.size() != static_cast<size_t>(n_nodes) ||
                node_depth.size() != static_cast<size_t>(n_nodes)) {
                throw std::invalid_argument("node arrays must have consistent length");
            }
            const int32_t n_links = static_cast<int32_t>(link_from.size());
            if (link_to.size() != static_cast<size_t>(n_links) ||
                link_length.size() != static_cast<size_t>(n_links) ||
                link_roughness_n.size() != static_cast<size_t>(n_links) ||
                link_diameter.size() != static_cast<size_t>(n_links) ||
                link_max_flow.size() != static_cast<size_t>(n_links) ||
                link_flow.size() != static_cast<size_t>(n_links)) {
                throw std::invalid_argument("link arrays must have consistent length");
            }
            const int32_t n_inlets = static_cast<int32_t>(inlet_cell.size());
            if (inlet_node.size() != static_cast<size_t>(n_inlets) ||
                inlet_crest_elev.size() != static_cast<size_t>(n_inlets) ||
                inlet_width.size() != static_cast<size_t>(n_inlets) ||
                inlet_coefficient.size() != static_cast<size_t>(n_inlets) ||
                inlet_max_capture.size() != static_cast<size_t>(n_inlets)) {
                throw std::invalid_argument("inlet arrays must have consistent length");
            }
            const int32_t n_outfalls = static_cast<int32_t>(outfall_cell.size());
            if (outfall_node.size() != static_cast<size_t>(n_outfalls) ||
                outfall_invert_elev.size() != static_cast<size_t>(n_outfalls) ||
                outfall_diameter.size() != static_cast<size_t>(n_outfalls) ||
                outfall_coefficient.size() != static_cast<size_t>(n_outfalls) ||
                outfall_max_flow.size() != static_cast<size_t>(n_outfalls) ||
                outfall_zero_storage.size() != static_cast<size_t>(n_outfalls)) {
                throw std::invalid_argument("outfall arrays must have consistent length");
            }
            const int32_t n_pipe_ends = static_cast<int32_t>(pipe_end_cell.size());
            if (pipe_end_node.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_invert_elev.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_diameter.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_area.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_inlet_loss_k.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_outlet_loss_k.size() != static_cast<size_t>(n_pipe_ends)) {
                throw std::invalid_argument("pipe_end arrays must have consistent length");
            }

            auto node_depth_out = py::array_t<double>(n_nodes);
            auto link_flow_out = py::array_t<double>(n_links);
            auto q_cell_out = py::array_t<double>(n_cells);
            double max_node_depth = 0.0;
            double max_link_flow = 0.0;
            double limiter_events = 0.0;
            double limiter_volume_m3 = 0.0;

            swe2d_gpu_drainage_step(
                n_cells,
                n_nodes,
                n_links,
                n_inlets,
                n_outfalls,
                n_pipe_ends,
                cell_wse.data(),
                cell_area.data(),
                node_invert_elev.data(),
                node_max_depth.data(),
                node_surface_area.data(),
                link_from.data(),
                link_to.data(),
                link_length.data(),
                link_roughness_n.data(),
                link_diameter.data(),
                link_max_flow.data(),
                inlet_cell.data(),
                inlet_node.data(),
                inlet_crest_elev.data(),
                inlet_width.data(),
                inlet_coefficient.data(),
                inlet_max_capture.data(),
                outfall_cell.data(),
                outfall_node.data(),
                outfall_invert_elev.data(),
                outfall_diameter.data(),
                outfall_coefficient.data(),
                outfall_max_flow.data(),
                outfall_zero_storage.data(),
                pipe_end_cell.data(),
                pipe_end_node.data(),
                pipe_end_invert_elev.data(),
                pipe_end_diameter.data(),
                pipe_end_area.data(),
                pipe_end_inlet_loss_k.data(),
                pipe_end_outlet_loss_k.data(),
                (cell_depth.size() == static_cast<size_t>(n_cells)) ? cell_depth.data() : nullptr,
                node_depth.data(),
                link_flow.data(),
                dt_s,
                gravity,
                solver_mode,
                head_deadband_m,
                dynamic_flow_relaxation,
                node_depth_out.mutable_data(),
                link_flow_out.mutable_data(),
                q_cell_out.mutable_data(),
                &max_node_depth,
                &max_link_flow,
                &limiter_events,
                &limiter_volume_m3);

            py::dict diag;
            diag["max_node_depth"] = max_node_depth;
            diag["max_link_flow"] = max_link_flow;
            diag["limiter_events"] = limiter_events;
            diag["limiter_volume_m3"] = limiter_volume_m3;
            return py::make_tuple(node_depth_out, link_flow_out, q_cell_out, diag);
        },
        py::arg("cell_wse"),
        py::arg("cell_area"),
        py::arg("node_invert_elev"),
        py::arg("node_max_depth"),
        py::arg("node_surface_area"),
        py::arg("link_from"),
        py::arg("link_to"),
        py::arg("link_length"),
        py::arg("link_roughness_n"),
        py::arg("link_diameter"),
        py::arg("link_max_flow"),
        py::arg("inlet_cell"),
        py::arg("inlet_node"),
        py::arg("inlet_crest_elev"),
        py::arg("inlet_width"),
        py::arg("inlet_coefficient"),
        py::arg("inlet_max_capture"),
        py::arg("outfall_cell"),
        py::arg("outfall_node"),
        py::arg("outfall_invert_elev"),
        py::arg("outfall_diameter"),
        py::arg("outfall_coefficient"),
        py::arg("outfall_max_flow"),
        py::arg("outfall_zero_storage"),
        py::arg("pipe_end_cell"),
        py::arg("pipe_end_node"),
        py::arg("pipe_end_invert_elev"),
        py::arg("pipe_end_diameter"),
        py::arg("pipe_end_area"),
        py::arg("pipe_end_inlet_loss_k"),
        py::arg("pipe_end_outlet_loss_k"),
        py::arg("cell_depth"),
        py::arg("node_depth"),
        py::arg("link_flow"),
        py::arg("dt_s"),
        py::arg("gravity"),
        py::arg("solver_mode"),
        py::arg("head_deadband_m") = 1.0e-3,
        py::arg("dynamic_flow_relaxation") = 1.0,
        "Headless CUDA helper: advance 1D drainage network one step (EGL/diffusion/dynamic).");

    m.def("swe2d_gpu_drainage_step_iterative",
        [](py::array_t<double, py::array::c_style | py::array::forcecast> cell_bed,
           py::array_t<double, py::array::c_style | py::array::forcecast> cell_area,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_invert_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_max_depth,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_surface_area,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> link_from,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> link_to,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_length,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_roughness_n,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_diameter,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_max_flow,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> inlet_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> inlet_node,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_crest_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_width,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_coefficient,
           py::array_t<double, py::array::c_style | py::array::forcecast> inlet_max_capture,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> outfall_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> outfall_node,
           py::array_t<double, py::array::c_style | py::array::forcecast> outfall_invert_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> outfall_diameter,
           py::array_t<double, py::array::c_style | py::array::forcecast> outfall_coefficient,
           py::array_t<double, py::array::c_style | py::array::forcecast> outfall_max_flow,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> outfall_zero_storage,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> pipe_end_cell,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast> pipe_end_node,
           py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_invert_elev,
           py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_diameter,
           py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_area,
           py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_inlet_loss_k,
           py::array_t<double, py::array::c_style | py::array::forcecast> pipe_end_outlet_loss_k,
           py::array_t<double, py::array::c_style | py::array::forcecast> cell_depth,
           py::array_t<double, py::array::c_style | py::array::forcecast> node_depth,
           py::array_t<double, py::array::c_style | py::array::forcecast> link_flow,
           double dt_s,
           double gravity,
           int32_t solver_mode,
           double head_deadband_m,
           double dynamic_flow_relaxation,
           int32_t n_substeps,
           int32_t implicit_iters,
           double coupling_relaxation)
           -> py::tuple
        {
            const int32_t n_cells = static_cast<int32_t>(cell_bed.size());
            if (cell_area.size() != static_cast<size_t>(n_cells) ||
                cell_depth.size() != static_cast<size_t>(n_cells)) {
                throw std::invalid_argument("cell_bed, cell_area, and cell_depth must have same length");
            }
            const int32_t n_nodes = static_cast<int32_t>(node_invert_elev.size());
            if (node_max_depth.size() != static_cast<size_t>(n_nodes) ||
                node_surface_area.size() != static_cast<size_t>(n_nodes) ||
                node_depth.size() != static_cast<size_t>(n_nodes)) {
                throw std::invalid_argument("node arrays must have consistent length");
            }
            const int32_t n_links = static_cast<int32_t>(link_from.size());
            if (link_to.size() != static_cast<size_t>(n_links) ||
                link_length.size() != static_cast<size_t>(n_links) ||
                link_roughness_n.size() != static_cast<size_t>(n_links) ||
                link_diameter.size() != static_cast<size_t>(n_links) ||
                link_max_flow.size() != static_cast<size_t>(n_links) ||
                link_flow.size() != static_cast<size_t>(n_links)) {
                throw std::invalid_argument("link arrays must have consistent length");
            }
            const int32_t n_inlets = static_cast<int32_t>(inlet_cell.size());
            if (inlet_node.size() != static_cast<size_t>(n_inlets) ||
                inlet_crest_elev.size() != static_cast<size_t>(n_inlets) ||
                inlet_width.size() != static_cast<size_t>(n_inlets) ||
                inlet_coefficient.size() != static_cast<size_t>(n_inlets) ||
                inlet_max_capture.size() != static_cast<size_t>(n_inlets)) {
                throw std::invalid_argument("inlet arrays must have consistent length");
            }
            const int32_t n_outfalls = static_cast<int32_t>(outfall_cell.size());
            if (outfall_node.size() != static_cast<size_t>(n_outfalls) ||
                outfall_invert_elev.size() != static_cast<size_t>(n_outfalls) ||
                outfall_diameter.size() != static_cast<size_t>(n_outfalls) ||
                outfall_coefficient.size() != static_cast<size_t>(n_outfalls) ||
                outfall_max_flow.size() != static_cast<size_t>(n_outfalls) ||
                outfall_zero_storage.size() != static_cast<size_t>(n_outfalls)) {
                throw std::invalid_argument("outfall arrays must have consistent length");
            }
            const int32_t n_pipe_ends = static_cast<int32_t>(pipe_end_cell.size());
            if (pipe_end_node.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_invert_elev.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_diameter.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_area.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_inlet_loss_k.size() != static_cast<size_t>(n_pipe_ends) ||
                pipe_end_outlet_loss_k.size() != static_cast<size_t>(n_pipe_ends)) {
                throw std::invalid_argument("pipe_end arrays must have consistent length");
            }

            const int32_t substeps = std::max<int32_t>(1, n_substeps);
            const int32_t iters = std::max<int32_t>(1, implicit_iters);
            const double relax = std::max(0.0, std::min(1.0, coupling_relaxation));
            const double dt_sub = dt_s / static_cast<double>(substeps);

            auto node_state_out = py::array_t<double>(n_nodes);
            auto link_state_out = py::array_t<double>(n_links);
            auto q_cell_out = py::array_t<double>(n_cells);

            std::vector<double> node_state(static_cast<size_t>(n_nodes));
            std::vector<double> link_state(static_cast<size_t>(n_links));
            std::vector<double> q_cell_acc(static_cast<size_t>(n_cells), 0.0);
            std::vector<double> q_cell_last(static_cast<size_t>(n_cells), 0.0);
            std::vector<double> hh_sub(static_cast<size_t>(n_cells));
            std::vector<double> hh_iter(static_cast<size_t>(n_cells));
            std::vector<double> hh_target(static_cast<size_t>(n_cells));
            std::vector<double> cell_wse(static_cast<size_t>(n_cells));
            std::vector<double> node_out(static_cast<size_t>(n_nodes));
            std::vector<double> link_out(static_cast<size_t>(n_links));
            std::vector<double> q_out(static_cast<size_t>(n_cells));

            std::memcpy(node_state.data(), node_depth.data(), sizeof(double) * static_cast<size_t>(n_nodes));
            std::memcpy(link_state.data(), link_flow.data(), sizeof(double) * static_cast<size_t>(n_links));
            std::memcpy(hh_sub.data(), cell_depth.data(), sizeof(double) * static_cast<size_t>(n_cells));

            // Fast path for fully inactive exchange states.
            const double tiny_h = std::max(1.0e-12, 0.1 * head_deadband_m);
            bool any_wet_surface = false;
            for (int32_t i = 0; i < n_cells; ++i) {
                if (hh_sub[static_cast<size_t>(i)] > tiny_h) {
                    any_wet_surface = true;
                    break;
                }
            }
            bool any_wet_nodes = false;
            for (int32_t i = 0; i < n_nodes; ++i) {
                if (node_state[static_cast<size_t>(i)] > tiny_h) {
                    any_wet_nodes = true;
                    break;
                }
            }
            bool any_link_flow = false;
            for (int32_t i = 0; i < n_links; ++i) {
                if (std::abs(link_state[static_cast<size_t>(i)]) > 1.0e-10) {
                    any_link_flow = true;
                    break;
                }
            }
            if (!any_wet_surface && !any_wet_nodes && !any_link_flow) {
                std::memcpy(node_state_out.mutable_data(), node_state.data(), sizeof(double) * static_cast<size_t>(n_nodes));
                std::memcpy(link_state_out.mutable_data(), link_state.data(), sizeof(double) * static_cast<size_t>(n_links));
                std::fill(q_cell_out.mutable_data(), q_cell_out.mutable_data() + n_cells, 0.0);
                py::dict diag;
                diag["max_node_depth"] = 0.0;
                diag["max_link_flow"] = 0.0;
                diag["limiter_events"] = 0.0;
                diag["limiter_volume_m3"] = 0.0;
                diag["substeps_used"] = 0.0;
                diag["implicit_iters_used"] = 0.0;
                diag["inactive_fastpath"] = 1.0;
                return py::make_tuple(node_state_out, link_state_out, q_cell_out, diag);
            }

            double max_node_depth = 0.0;
            double max_link_flow = 0.0;
            double limiter_events = 0.0;
            double limiter_volume_m3 = 0.0;

            int32_t substeps_used = 0;
            int32_t implicit_iters_used = 0;
            for (int32_t s = 0; s < substeps; ++s) {
                ++substeps_used;
                std::memcpy(hh_iter.data(), hh_sub.data(), sizeof(double) * static_cast<size_t>(n_cells));
                std::fill(q_cell_last.begin(), q_cell_last.end(), 0.0);
                double step_max_node_depth = 0.0;
                double step_max_link_flow = 0.0;
                double step_limiter_events = 0.0;
                double step_limiter_volume_m3 = 0.0;
                bool converged = false;

                for (int32_t it = 0; it < iters; ++it) {
                    ++implicit_iters_used;
                    for (int32_t i = 0; i < n_cells; ++i) {
                        cell_wse[static_cast<size_t>(i)] = cell_bed.data()[i] + hh_iter[static_cast<size_t>(i)];
                    }
                    swe2d_gpu_drainage_step(
                        n_cells,
                        n_nodes,
                        n_links,
                        n_inlets,
                        n_outfalls,
                        n_pipe_ends,
                        cell_wse.data(),
                        cell_area.data(),
                        node_invert_elev.data(),
                        node_max_depth.data(),
                        node_surface_area.data(),
                        link_from.data(),
                        link_to.data(),
                        link_length.data(),
                        link_roughness_n.data(),
                        link_diameter.data(),
                        link_max_flow.data(),
                        inlet_cell.data(),
                        inlet_node.data(),
                        inlet_crest_elev.data(),
                        inlet_width.data(),
                        inlet_coefficient.data(),
                        inlet_max_capture.data(),
                        outfall_cell.data(),
                        outfall_node.data(),
                        outfall_invert_elev.data(),
                        outfall_diameter.data(),
                        outfall_coefficient.data(),
                        outfall_max_flow.data(),
                        outfall_zero_storage.data(),
                        pipe_end_cell.data(),
                        pipe_end_node.data(),
                        pipe_end_invert_elev.data(),
                        pipe_end_diameter.data(),
                        pipe_end_area.data(),
                        pipe_end_inlet_loss_k.data(),
                        pipe_end_outlet_loss_k.data(),
                        hh_iter.data(),
                        node_state.data(),
                        link_state.data(),
                        dt_sub,
                        gravity,
                        solver_mode,
                        head_deadband_m,
                        dynamic_flow_relaxation,
                        node_out.data(),
                        link_out.data(),
                        q_out.data(),
                        &step_max_node_depth,
                        &step_max_link_flow,
                        &step_limiter_events,
                        &step_limiter_volume_m3);

                    std::memcpy(node_state.data(), node_out.data(), sizeof(double) * static_cast<size_t>(n_nodes));
                    std::memcpy(link_state.data(), link_out.data(), sizeof(double) * static_cast<size_t>(n_links));
                    std::memcpy(q_cell_last.data(), q_out.data(), sizeof(double) * static_cast<size_t>(n_cells));

                    double max_h_update = 0.0;
                    for (int32_t i = 0; i < n_cells; ++i) {
                        const double h_prev = hh_iter[static_cast<size_t>(i)];
                        const double h_old = hh_sub[static_cast<size_t>(i)];
                        const double dh = q_cell_last[static_cast<size_t>(i)] * dt_sub / std::max(cell_area.data()[i], 1.0e-12);
                        const double h_tgt = std::max(h_old - dh, 0.0);
                        hh_target[static_cast<size_t>(i)] = h_tgt;
                        hh_iter[static_cast<size_t>(i)] = (1.0 - relax) * h_prev + relax * h_tgt;
                        max_h_update = std::max(max_h_update, std::abs(hh_iter[static_cast<size_t>(i)] - h_prev));
                    }

                    if (max_h_update <= tiny_h) {
                        converged = true;
                        break;
                    }
                }

                for (int32_t i = 0; i < n_cells; ++i) {
                    q_cell_acc[static_cast<size_t>(i)] += q_cell_last[static_cast<size_t>(i)];
                    const double dh = q_cell_last[static_cast<size_t>(i)] * dt_sub / std::max(cell_area.data()[i], 1.0e-12);
                    hh_sub[static_cast<size_t>(i)] = std::max(hh_sub[static_cast<size_t>(i)] - dh, 0.0);
                }
                max_node_depth = std::max(max_node_depth, step_max_node_depth);
                max_link_flow = std::max(max_link_flow, step_max_link_flow);
                limiter_events += step_limiter_events;
                limiter_volume_m3 += step_limiter_volume_m3;

                if (converged) {
                    break;
                }
            }

            for (int32_t i = 0; i < n_cells; ++i) {
                q_cell_out.mutable_data()[i] = q_cell_acc[static_cast<size_t>(i)] / static_cast<double>(std::max(1, substeps_used));
            }

            std::memcpy(node_state_out.mutable_data(), node_state.data(), sizeof(double) * static_cast<size_t>(n_nodes));
            std::memcpy(link_state_out.mutable_data(), link_state.data(), sizeof(double) * static_cast<size_t>(n_links));

            py::dict diag;
            diag["max_node_depth"] = max_node_depth;
            diag["max_link_flow"] = max_link_flow;
            diag["limiter_events"] = limiter_events;
            diag["limiter_volume_m3"] = limiter_volume_m3;
            diag["substeps_used"] = static_cast<double>(substeps_used);
            diag["implicit_iters_used"] = static_cast<double>(implicit_iters_used);
            diag["inactive_fastpath"] = 0.0;
            return py::make_tuple(node_state_out, link_state_out, q_cell_out, diag);
        },
        py::arg("cell_bed"),
        py::arg("cell_area"),
        py::arg("node_invert_elev"),
        py::arg("node_max_depth"),
        py::arg("node_surface_area"),
        py::arg("link_from"),
        py::arg("link_to"),
        py::arg("link_length"),
        py::arg("link_roughness_n"),
        py::arg("link_diameter"),
        py::arg("link_max_flow"),
        py::arg("inlet_cell"),
        py::arg("inlet_node"),
        py::arg("inlet_crest_elev"),
        py::arg("inlet_width"),
        py::arg("inlet_coefficient"),
        py::arg("inlet_max_capture"),
        py::arg("outfall_cell"),
        py::arg("outfall_node"),
        py::arg("outfall_invert_elev"),
        py::arg("outfall_diameter"),
        py::arg("outfall_coefficient"),
        py::arg("outfall_max_flow"),
        py::arg("outfall_zero_storage"),
        py::arg("pipe_end_cell"),
        py::arg("pipe_end_node"),
        py::arg("pipe_end_invert_elev"),
        py::arg("pipe_end_diameter"),
        py::arg("pipe_end_area"),
        py::arg("pipe_end_inlet_loss_k"),
        py::arg("pipe_end_outlet_loss_k"),
        py::arg("cell_depth"),
        py::arg("node_depth"),
        py::arg("link_flow"),
        py::arg("dt_s"),
        py::arg("gravity"),
        py::arg("solver_mode"),
        py::arg("head_deadband_m") = 1.0e-3,
        py::arg("dynamic_flow_relaxation") = 1.0,
        py::arg("n_substeps") = 1,
        py::arg("implicit_iters") = 1,
        py::arg("coupling_relaxation") = 0.5,
        "Headless CUDA helper: advance drainage network with native substep/implicit loops in one call.");

    m.def("swe2d_gpu_enable_kernel_graphs",
        [](py::object dev_capsule, bool enable) {
            auto dev = static_cast<SWE2DDeviceState*>(PyCapsule_GetPointer(dev_capsule.ptr(), "SWE2DDeviceState*"));
            if (!dev) throw std::runtime_error("Invalid device pointer");
            swe2d_gpu_enable_kernel_graphs(dev, enable);
        },
        py::arg("dev"),
        py::arg("enable"),
        "Enable or disable CUDA graph optimization for kernel sequence replay.");

    m.def("swe2d_gpu_destroy_kernel_graphs",
        [](py::object dev_capsule) {
            auto dev = static_cast<SWE2DDeviceState*>(PyCapsule_GetPointer(dev_capsule.ptr(), "SWE2DDeviceState*"));
            if (!dev) throw std::runtime_error("Invalid device pointer");
            swe2d_gpu_destroy_kernel_graphs(dev);
        },
        py::arg("dev"),
        "Destroy cached CUDA graph resources for this solver instance.");
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

    m.def("swe2d_gpu_drainage_step",
        [](py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           double,
           double,
           int32_t,
           double,
           double) -> py::tuple
        {
            throw std::runtime_error("CUDA path not compiled; swe2d_gpu_drainage_step is unavailable.");
        });

    m.def("swe2d_gpu_drainage_step_iterative",
        [](py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<int32_t, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           py::array_t<double, py::array::c_style | py::array::forcecast>,
           double,
           double,
           int32_t,
           double,
           double,
           int32_t,
           int32_t,
           double) -> py::tuple
        {
            throw std::runtime_error("CUDA path not compiled; swe2d_gpu_drainage_step_iterative is unavailable.");
        });

    m.def("swe2d_gpu_enable_kernel_graphs",
        [](py::object, bool) {
            throw std::runtime_error("CUDA path not compiled; swe2d_gpu_enable_kernel_graphs is unavailable.");
        });

    m.def("swe2d_gpu_destroy_kernel_graphs",
        [](py::object) {
            throw std::runtime_error("CUDA path not compiled; swe2d_gpu_destroy_kernel_graphs is unavailable.");
        });
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

    m.def("swe2d_solver_set_external_sources",
        [](const std::shared_ptr<PySolver>& ps,
           py::object source_mps_obj) {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");
            if (source_mps_obj.is_none()) {
                swe2d_solver_set_external_sources(ps->solver, nullptr, 0);
                return;
            }
            auto src = source_mps_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
            const int32_t nc = ps->solver->mesh->n_cells;
            if (src.size() != static_cast<size_t>(nc)) {
                throw std::invalid_argument("source_mps length must equal n_cells");
            }
            swe2d_solver_set_external_sources(ps->solver, src.data(), nc);
        },
        py::arg("solver"), py::arg("source_mps") = py::none(),
        "Set per-cell external depth source rates [m/s] on solver (None clears).");

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
                  bool extreme_rain_mode,
                  double source_cfl_beta,
                  int source_max_substeps,
                  double source_rate_cap,
                  double source_depth_step_cap,
                  bool source_true_subcycling,
                  bool source_imex_split,
                  bool enable_shallow_front_recon_fallback,
                  int gpu_diag_sync_interval_steps,
              bool use_gpu, int n_threads,
              int temporal_order,
              int spatial_scheme,
              int godunov_mode,
              int turbulence_model,
              int bed_friction_model,
              int equation_set,
              int coupling_mode,
              int three_d_solver_model,
              bool enforce_gpu_only_advanced_modes,
              bool three_d_single_phase_free_surface,
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
            cfg.extreme_rain_mode = extreme_rain_mode;
            cfg.source_cfl_beta = source_cfl_beta;
            cfg.source_max_substeps = source_max_substeps;
            cfg.source_rate_cap = source_rate_cap;
            cfg.source_depth_step_cap = source_depth_step_cap;
            cfg.source_true_subcycling = source_true_subcycling;
            cfg.source_imex_split = source_imex_split;
            cfg.enable_shallow_front_recon_fallback = enable_shallow_front_recon_fallback;
            cfg.gpu_diag_sync_interval_steps = gpu_diag_sync_interval_steps;
            cfg.temporal_order = temporal_order;
            cfg.spatial_scheme = spatial_scheme;
            cfg.godunov_mode = godunov_mode;
            cfg.turbulence_model = turbulence_model;
            cfg.bed_friction_model = bed_friction_model;
            cfg.equation_set = equation_set;
            cfg.coupling_mode = coupling_mode;
            cfg.three_d_solver_model = three_d_solver_model;
            cfg.enforce_gpu_only_advanced_modes = enforce_gpu_only_advanced_modes;
            cfg.three_d_single_phase_free_surface = three_d_single_phase_free_surface;
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
        py::arg("extreme_rain_mode") = false,
        py::arg("source_cfl_beta") = 0.25,
        py::arg("source_max_substeps") = 16,
        py::arg("source_rate_cap") = 0.0,
        py::arg("source_depth_step_cap") = 0.0,
        py::arg("source_true_subcycling") = false,
        py::arg("source_imex_split") = false,
        py::arg("enable_shallow_front_recon_fallback") = true,
        py::arg("gpu_diag_sync_interval_steps") = 1,
        py::arg("use_gpu")  = true,
        py::arg("n_threads") = 0,
        py::arg("temporal_order") = 2,
        py::arg("spatial_scheme") = 0,
        py::arg("godunov_mode") = 0,
        py::arg("turbulence_model") = 0,
        py::arg("bed_friction_model") = 0,
        py::arg("equation_set") = 0,
        py::arg("coupling_mode") = 0,
        py::arg("three_d_solver_model") = 0,
        py::arg("enforce_gpu_only_advanced_modes") = true,
        py::arg("three_d_single_phase_free_surface") = true,
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
            d["gpu_graph_launches_step"] = diag.gpu_graph_launches_step;
            d["gpu_graph_launches_total"] = diag.gpu_graph_launches_total;
            d["projection_retry_count"] = diag.projection_retry_count;
            d["projection_attempt_count"] = diag.projection_attempt_count;
            d["projection_retry_exhausted"] = diag.projection_retry_exhausted;
            d["projection_retry_enabled"] = diag.projection_retry_enabled;
            d["projection_retry_fail_fast"] = diag.projection_retry_fail_fast;
            d["projection_retry_dt_initial"] = diag.projection_retry_dt_initial;
            d["projection_retry_dt_floor"] = diag.projection_retry_dt_floor;
            d["projection_retry_dt_reduction"] = diag.projection_retry_dt_reduction;
            d["projection_retry_residual_target"] = diag.projection_retry_residual_target;
            d["projection_retry_residual_ratio"] = diag.projection_retry_residual_ratio;
            d["projection_retry_residual_ratio_max"] = diag.projection_retry_residual_ratio_max;
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

    // ── Native run-to-time loop ───────────────────────────────────────────────
    m.def("swe2d_run_to_time",
        [](std::shared_ptr<PySolver> ps,
           double t_end,
           double dt_request,
           int diag_batch_size) -> py::dict
        {
            if (!ps || !ps->solver) throw std::invalid_argument("null solver handle");

            // Run the native loop without Python callbacks (callbacks would require
            // a context pointer in the C interface, which we don't have).
            // For now, we batch diagnostics and return them after completion.
            std::vector<SWE2DStepDiag> diag_batch;
            if (diag_batch_size > 0) {
                diag_batch.reserve(diag_batch_size);
            }

            SWE2DRunConfig cfg;
            cfg.t_end = t_end;
            cfg.dt_request = dt_request;
            cfg.progress_callback_interval_steps = 0;  // No Python callbacks
            cfg.progress_cb = nullptr;
            cfg.diag_batch_size = diag_batch_size;
            cfg.progress_callback_interval_steps = 0;
            cfg.progress_cb = nullptr;

            // Allocate temp array for diagnostics if batching enabled.
            std::vector<SWE2DStepDiag> temp_diag_array;
            if (diag_batch_size > 0) {
                temp_diag_array.resize(diag_batch_size);
            }

            int32_t result = swe2d_run_to_time(
                ps->solver,
                &cfg,
                temp_diag_array.size() > 0 ? temp_diag_array.data() : nullptr,
                static_cast<int32_t>(temp_diag_array.size()));

            // Convert diagnostics to Python list.
            py::list diag_list;
            if (result > 0) {
                for (int32_t i = 0; i < result; ++i) {
                    const SWE2DStepDiag& d = temp_diag_array[i];
                    py::dict d_dict;
                    d_dict["dt"] = d.dt;
                    d_dict["wet_cells"] = static_cast<int32_t>(d.wet_cells);
                    d_dict["max_depth"] = d.max_depth;
                    d_dict["min_depth"] = d.min_depth;
                    d_dict["mass_total"] = d.mass_total;
                    d_dict["max_courant"] = d.max_courant;
                    d_dict["max_depth_residual"] = d.max_depth_residual;
                    d_dict["max_wse_elev_error"] = d.max_wse_elev_error;
                    d_dict["gpu_active"] = d.gpu_active;
                    d_dict["gpu_graph_launches_step"] = static_cast<int32_t>(d.gpu_graph_launches_step);
                    d_dict["projection_retry_count"] = d.projection_retry_count;
                    d_dict["projection_attempt_count"] = d.projection_attempt_count;
                    d_dict["projection_retry_exhausted"] = d.projection_retry_exhausted;
                    d_dict["projection_retry_enabled"] = d.projection_retry_enabled;
                    d_dict["projection_retry_fail_fast"] = d.projection_retry_fail_fast;
                    d_dict["projection_retry_dt_initial"] = d.projection_retry_dt_initial;
                    d_dict["projection_retry_dt_floor"] = d.projection_retry_dt_floor;
                    d_dict["projection_retry_dt_reduction"] = d.projection_retry_dt_reduction;
                    d_dict["projection_retry_residual_target"] = d.projection_retry_residual_target;
                    d_dict["projection_retry_residual_ratio"] = d.projection_retry_residual_ratio;
                    d_dict["projection_retry_residual_ratio_max"] = d.projection_retry_residual_ratio_max;
                    diag_list.append(d_dict);
                }
            }

            py::dict ret;
            ret["diags"] = diag_list;
            ret["steps_completed"] = static_cast<int32_t>(std::abs(result));
            ret["cancelled"] = (result < 0);
            ret["final_time"] = ps->solver->t;
            return ret;
        },
        py::arg("solver"),
        py::arg("t_end"),
        py::arg("dt_request") = -1.0,
        py::arg("diag_batch_size") = 0,
        "Run simulation natively from current time to t_end. Returns dict with 'diags', "
        "'steps_completed', 'cancelled', 'final_time'.");

    // ─────────────────────────────────────────────────────────────────────────────
    // Phase 7: 2D-3D interface contract API
    // ─────────────────────────────────────────────────────────────────────────────

    // Wrapper class for contract handle (Python GC owns lifetime)
    struct PyContractHandle {
        SWE2D3DInterfaceContractHost host_contract;
        // Device contract (if uploaded) is managed by solver, not this handle.
    };

    // Create contract from arrays (validates and deep-copies).
    m.def("swe2d_contract_create",
        [](py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell2d,
           py::array_t<double, py::array::c_style | py::array::forcecast> face_area,
           py::array_t<double, py::array::c_style | py::array::forcecast> face_nx,
           py::array_t<double, py::array::c_style | py::array::forcecast> face_ny,
           py::array_t<double, py::array::c_style | py::array::forcecast> face_nz) 
            -> std::shared_ptr<PyContractHandle>
        {
            auto handle = std::make_shared<PyContractHandle>();
            
            // Copy arrays into host contract
            const int32_t* c2d_ptr = cell2d.data();
            const double* fa_ptr = face_area.data();
            const double* fnx_ptr = face_nx.data();
            const double* fny_ptr = face_ny.data();
            const double* fnz_ptr = face_nz.data();
            
            int32_t n = static_cast<int32_t>(cell2d.size());
            if (n <= 0 ||
                face_area.size() != n ||
                face_nx.size() != n ||
                face_ny.size() != n ||
                face_nz.size() != n) {
                throw std::invalid_argument(
                    "swe2d_contract_create: all arrays must have same length > 0");
            }
            
            handle->host_contract.cell2d.assign(c2d_ptr, c2d_ptr + n);
            handle->host_contract.face_area.assign(fa_ptr, fa_ptr + n);
            handle->host_contract.face_nx.assign(fnx_ptr, fnx_ptr + n);
            handle->host_contract.face_ny.assign(fny_ptr, fny_ptr + n);
            handle->host_contract.face_nz.assign(fnz_ptr, fnz_ptr + n);
            
            return handle;
        },
        py::arg("cell2d"), py::arg("face_area"), py::arg("face_nx"),
        py::arg("face_ny"), py::arg("face_nz"),
        "Create a 2D-3D interface contract from numpy arrays. Arrays must all have same length.");

    // Validate contract before upload.
    m.def("swe2d_contract_is_valid",
        [](const std::shared_ptr<PyContractHandle>& contract) -> bool
        {
            if (!contract) return false;
            return swe2d_contract_is_valid(contract->host_contract);
        },
        py::arg("contract"),
        "Validate contract consistency (all arrays same length, non-empty).");

    // Upload contract to GPU solver.
    m.def("swe2d_gpu_contract_upload",
        [](const std::shared_ptr<PySolver>& solver,
           const std::shared_ptr<PyContractHandle>& contract) -> bool
        {
            if (!solver || !solver->solver) 
                throw std::invalid_argument("null solver handle");
            if (!contract) 
                throw std::invalid_argument("null contract handle");
            
            #ifdef BACKWATER_HAS_CUDA
            return swe2d_gpu_contract_upload(solver->solver->dev, contract->host_contract);
            #else
            throw std::runtime_error("GPU support not compiled; cannot upload contract");
            #endif
        },
        py::arg("solver"), py::arg("contract"),
        "Upload contract geometry and allocate device buffers for exchange. Returns True on success.");

    // Clear contract (free device buffers).
    m.def("swe2d_gpu_contract_clear",
        [](const std::shared_ptr<PySolver>& solver) -> void
        {
            if (!solver || !solver->solver) 
                throw std::invalid_argument("null solver handle");
            
            #ifdef BACKWATER_HAS_CUDA
            swe2d_gpu_contract_free(solver->solver->dev);
            #endif
        },
        py::arg("solver"),
        "Clear device-side contract buffers (flux, head-loss, etc).");

    // Query: is contract uploaded?
    m.def("swe2d_gpu_is_contract_uploaded",
        [](const std::shared_ptr<PySolver>& solver) -> bool
        {
            if (!solver || !solver->solver) return false;
            
            #ifdef BACKWATER_HAS_CUDA
            return swe2d_gpu_is_contract_uploaded(solver->solver->dev);
            #else
            return false;
            #endif
        },
        py::arg("solver"),
        "Return True if a device contract is currently uploaded.");

    // ── 3D patch state observation/initialisation (validation API) ────────────

    m.def("swe2d_get_3d_patch_stats",
        [](const std::shared_ptr<PySolver>& solver) -> py::dict
        {
            if (!solver || !solver->solver)
                throw std::invalid_argument("null or destroyed solver");
            #ifdef BACKWATER_HAS_CUDA
            SWE3DPatchStats s = swe2d_gpu_get_3d_patch_stats(solver->solver->dev);
            py::dict d;
            d["n_cells"]   = s.n_cells;
            d["nx"]        = s.nx;
            d["ny"]        = s.ny;
            d["nz"]        = s.nz;
            d["dx"]        = s.dx;
            d["dy"]        = s.dy;
            d["dz"]        = s.dz;
            d["vof_min"]   = s.vof_min;
            d["vof_max"]   = s.vof_max;
            d["vof_sum"]   = s.vof_sum;
            d["u_rms"]     = s.u_rms;
            d["v_rms"]     = s.v_rms;
            d["w_rms"]     = s.w_rms;
            d["p_max_abs"] = s.p_max_abs;
            d["divergence_rms"] = s.divergence_rms;
            d["projection_iters"] = s.projection_iters;
            d["projection_residual"] = s.projection_residual;
            d["projection_converged"] = s.projection_converged;
            d["vof_transport_substeps"] = s.vof_transport_substeps;
            return d;
            #else
            throw std::runtime_error("CUDA not compiled; swe2d_get_3d_patch_stats unavailable");
            #endif
        },
        py::arg("solver"),
        "Return aggregate statistics for the 3D Cartesian patch attached to solver.\n"
        "Performs a device→host transfer; intended for testing, not production inner loops.\n"
        "Keys: n_cells, nx, ny, nz, dx, dy, dz, vof_min, vof_max, vof_sum,\n"
        "      u_rms, v_rms, w_rms, p_max_abs, divergence_rms,\n"
        "      projection_iters, projection_residual, projection_converged,\n"
        "      vof_transport_substeps");

    m.def("swe2d_set_3d_patch_vof",
        [](const std::shared_ptr<PySolver>& solver, py::array_t<double, py::array::c_style> vof) -> void
        {
            if (!solver || !solver->solver)
                throw std::invalid_argument("null or destroyed solver");
            #ifdef BACKWATER_HAS_CUDA
            py::buffer_info buf = vof.request();
            swe2d_gpu_set_3d_patch_vof(
                solver->solver->dev,
                static_cast<const double*>(buf.ptr),
                static_cast<int64_t>(buf.size));
            #else
            throw std::runtime_error("CUDA not compiled; swe2d_set_3d_patch_vof unavailable");
            #endif
        },
        py::arg("solver"),
        py::arg("vof"),
        "Upload a VoF initial-condition array (float64, length == n_cells) to the 3D patch.");

    m.def("swe2d_get_3d_patch_vof",
        [](const std::shared_ptr<PySolver>& solver) -> py::array_t<double>
        {
            if (!solver || !solver->solver)
                throw std::invalid_argument("null or destroyed solver");
            #ifdef BACKWATER_HAS_CUDA
            SWE3DPatchStats s = swe2d_gpu_get_3d_patch_stats(solver->solver->dev);
            py::array_t<double> out(s.n_cells);
            py::buffer_info buf = out.request();
            swe2d_gpu_get_3d_patch_vof(
                solver->solver->dev,
                static_cast<double*>(buf.ptr),
                s.n_cells);
            return out;
            #else
            throw std::runtime_error("CUDA not compiled; swe2d_get_3d_patch_vof unavailable");
            #endif
        },
        py::arg("solver"),
        "Download full VoF field from 3D patch as a float64 array of length n_cells.");

    m.def("swe2d_set_3d_patch_state",
        [](const std::shared_ptr<PySolver>& solver,
           py::object u_obj,
           py::object v_obj,
           py::object w_obj,
           py::object p_obj,
           py::object vof_obj) -> void
        {
            if (!solver || !solver->solver)
                throw std::invalid_argument("null or destroyed solver");
            #ifdef BACKWATER_HAS_CUDA
            auto to_ptr = [](py::object& o, std::vector<double>& tmp) -> const double*
            {
                if (o.is_none()) return nullptr;
                auto arr = o.cast<py::array_t<double, py::array::c_style>>();
                py::buffer_info buf = arr.request();
                tmp.assign(static_cast<const double*>(buf.ptr),
                            static_cast<const double*>(buf.ptr) + buf.size);
                return tmp.data();
            };
            // infer n from first non-None arg
            int64_t n = -1;
            auto check_n = [&](py::object& o) {
                if (!o.is_none() && n < 0) {
                    auto arr = o.cast<py::array_t<double>>();
                    n = static_cast<int64_t>(arr.size());
                }
            };
            check_n(u_obj); check_n(v_obj); check_n(w_obj);
            check_n(p_obj); check_n(vof_obj);
            if (n < 0) return; // all None — no-op

            std::vector<double> tu, tv, tw, tp, tvof;
            const double* pu   = to_ptr(u_obj,   tu);
            const double* pv   = to_ptr(v_obj,   tv);
            const double* pw   = to_ptr(w_obj,   tw);
            const double* pp   = to_ptr(p_obj,   tp);
            const double* pvof = to_ptr(vof_obj, tvof);
            swe2d_gpu_set_3d_patch_state(solver->solver->dev, pu, pv, pw, pp, pvof, n);
            #else
            throw std::runtime_error("CUDA not compiled; swe2d_set_3d_patch_state unavailable");
            #endif
        },
        py::arg("solver"),
        py::arg("u")   = py::none(),
        py::arg("v")   = py::none(),
        py::arg("w")   = py::none(),
        py::arg("p")   = py::none(),
        py::arg("vof") = py::none(),
        "Upload per-cell initial conditions for any combination of u, v, w, p, vof.\n"
        "Pass None to skip a field.  Arrays must be float64, length == n_cells.");

    m.def("swe2d_set_3d_patch_geometry",
        [](const std::shared_ptr<PySolver>& solver,
           py::object phi_obj,
           py::object ax_obj,
           py::object ay_obj,
           py::object az_obj) -> void
        {
            if (!solver || !solver->solver)
                throw std::invalid_argument("null or destroyed solver");
            #ifdef BACKWATER_HAS_CUDA
            auto infer_n = [](py::object& o) -> int64_t {
                if (o.is_none()) return -1;
                auto arr = o.cast<py::array_t<double, py::array::c_style>>();
                return static_cast<int64_t>(arr.size());
            };

            int64_t n = -1;
            for (py::object* obj : {&phi_obj, &ax_obj, &ay_obj, &az_obj}) {
                const int64_t m = infer_n(*obj);
                if (m < 0) continue;
                if (n < 0) {
                    n = m;
                } else if (m != n) {
                    throw std::invalid_argument("swe2d_set_3d_patch_geometry: all provided arrays must have equal length");
                }
            }
            if (n < 0) return; // all None — no-op

            auto to_ptr = [n](py::object& o, std::vector<double>& tmp) -> const double*
            {
                if (o.is_none()) return nullptr;
                auto arr = o.cast<py::array_t<double, py::array::c_style>>();
                py::buffer_info buf = arr.request();
                if (static_cast<int64_t>(buf.size) != n) {
                    throw std::invalid_argument("swe2d_set_3d_patch_geometry: length mismatch among provided arrays");
                }
                tmp.assign(static_cast<const double*>(buf.ptr),
                           static_cast<const double*>(buf.ptr) + buf.size);
                return tmp.data();
            };

            std::vector<double> tphi, tax, tay, taz;
            const double* pphi = to_ptr(phi_obj, tphi);
            const double* pax = to_ptr(ax_obj, tax);
            const double* pay = to_ptr(ay_obj, tay);
            const double* paz = to_ptr(az_obj, taz);
            swe2d_gpu_set_3d_patch_geometry(solver->solver->dev, pphi, pax, pay, paz, n);
            #else
            throw std::runtime_error("CUDA not compiled; swe2d_set_3d_patch_geometry unavailable");
            #endif
        },
        py::arg("solver"),
        py::arg("phi") = py::none(),
        py::arg("ax") = py::none(),
        py::arg("ay") = py::none(),
        py::arg("az") = py::none(),
        "Upload static 3D geometry tensors for sub-grid solids.\n"
        "Pass any combination of phi/ax/ay/az arrays (float64, equal length == n_cells).\n"
        "Pass None for fields that should remain unchanged.");

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

    py::class_<PyContractHandle, std::shared_ptr<PyContractHandle>>(m, "SWE2DContractHandle")
        .def("__repr__", [](const PyContractHandle& pc) {
            return "<SWE2DContractHandle n_faces=" + std::to_string(pc.host_contract.cell2d.size()) + ">";
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
    m.attr("BC_NORMAL_DEPTH_SLOPE") = py::int_(static_cast<int>(BCType::NORMAL_DEPTH_SLOPE));
}
