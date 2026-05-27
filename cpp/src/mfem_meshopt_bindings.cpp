#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

py::dict optimize_mesh_tmop(
    py::array_t<double, py::array::c_style | py::array::forcecast> node_x,
    py::array_t<double, py::array::c_style | py::array::forcecast> node_y,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell_face_offsets,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell_face_nodes,
    py::object cell_nodes = py::none(),
    py::object cell_type = py::none(),
    py::object region_id = py::none(),
    py::object target_size = py::none(),
    py::object arc_region_ids = py::none(),
    py::object arc_roles = py::none(),
    py::object arc_lines = py::none(),
    double quality_weight = 1.0,
    double boundary_fit_weight = 0.35,
    double interface_fit_weight = 0.25,
    int max_iterations = 25,
    double min_det_j = 1.0e-9,
    bool preserve_boundary = true,
    bool lock_boundary_nodes = true
) {
    // Beta bridge contract: keep topology stable and return pass-through mesh.
    // This allows strict MFEM post-opt flow wiring before full TMOP numerics are integrated.
    py::dict out;
    out["node_x"] = node_x;
    out["node_y"] = node_y;
    out["cell_face_offsets"] = cell_face_offsets;
    out["cell_face_nodes"] = cell_face_nodes;

    if (!cell_nodes.is_none()) {
        out["cell_nodes"] = cell_nodes;
    }
    if (!cell_type.is_none()) {
        out["cell_type"] = cell_type;
    }
    if (!region_id.is_none()) {
        out["region_id"] = region_id;
    }
    if (!target_size.is_none()) {
        out["target_size"] = target_size;
    }

    py::dict meta;
    meta["engine"] = "hydra_mfem_meshopt_beta_stub";
    meta["quality_weight"] = quality_weight;
    meta["boundary_fit_weight"] = boundary_fit_weight;
    meta["interface_fit_weight"] = interface_fit_weight;
    meta["max_iterations"] = max_iterations;
    meta["min_det_j"] = min_det_j;
    meta["preserve_boundary"] = preserve_boundary;
    meta["lock_boundary_nodes"] = lock_boundary_nodes;
    meta["arc_count"] = arc_region_ids.is_none() ? 0 : py::len(arc_region_ids);
    out["quality_summary"] = meta;
    return out;
}

py::dict optimize_mesh(
    py::array_t<double, py::array::c_style | py::array::forcecast> node_x,
    py::array_t<double, py::array::c_style | py::array::forcecast> node_y,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell_face_offsets,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> cell_face_nodes
) {
    return optimize_mesh_tmop(
        node_x,
        node_y,
        cell_face_offsets,
        cell_face_nodes
    );
}

} // namespace

PYBIND11_MODULE(hydra_mfem_meshopt, m) {
    m.doc() = "Beta MFEM TMOP mesh optimization bridge (stub)";

    m.def(
        "optimize_mesh_tmop",
        &optimize_mesh_tmop,
        py::arg("node_x"),
        py::arg("node_y"),
        py::arg("cell_face_offsets"),
        py::arg("cell_face_nodes"),
        py::arg("cell_nodes") = py::none(),
        py::arg("cell_type") = py::none(),
        py::arg("region_id") = py::none(),
        py::arg("target_size") = py::none(),
        py::arg("arc_region_ids") = py::none(),
        py::arg("arc_roles") = py::none(),
        py::arg("arc_lines") = py::none(),
        py::arg("quality_weight") = 1.0,
        py::arg("boundary_fit_weight") = 0.35,
        py::arg("interface_fit_weight") = 0.25,
        py::arg("max_iterations") = 25,
        py::arg("min_det_j") = 1.0e-9,
        py::arg("preserve_boundary") = true,
        py::arg("lock_boundary_nodes") = true
    );

    m.def(
        "optimize_mesh",
        &optimize_mesh,
        py::arg("node_x"),
        py::arg("node_y"),
        py::arg("cell_face_offsets"),
        py::arg("cell_face_nodes")
    );
}
