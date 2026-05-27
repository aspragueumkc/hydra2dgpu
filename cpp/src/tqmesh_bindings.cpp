/*
 * tqmesh_bindings.cpp
 * pybind11 wrapper around TQMesh advancing-front mesh generator.
 *
 * Exposed to Python as module "backwater_tqmesh".
 *
 * Primary function:
 *   generate_triangular_mesh(
 *       ext_verts      : list[tuple[float,float]]   exterior CCW ring
 *       ext_colors     : list[int]                  BC color per ext edge
 *       int_boundaries : list[list[tuple[float,float]]]  interior CW holes
 *       int_colors     : list[list[int]]             BC color per int edge
 *       constraint_zones : list[dict]               refinement zones
 *           {verts: [(x,y)...], size: float}
 *       fixed_edges    : list[list[tuple[float,float]]]  fixed interior polylines
 *       target_size    : float                      global element size
 *       quad_layers    : list[(x0,y0,x1,y1,n,h,g)] optional quad-layer controls
 *       tri_to_quad    : bool                       apply tri-to-quad conversion
 *       n_smooth       : int  = 3                   Laplace smoothing passes
 *   ) -> dict with keys:
 *       verts_x, verts_y : float64 arrays  (N_v,)
 *       triangles        : int32 array     (N_t, 3)  node indices
 *       quads            : int32 array     (N_q, 4)  node indices
 *       bdry_v0, bdry_v1 : int32 arrays   (N_be,)   boundary edge endpoints
 *       bdry_color       : int32 array    (N_be,)    boundary edge BC color
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// Suppress all TQMesh internal warnings — it's a vendored lib
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"

#include "TQMesh.h"

#pragma GCC diagnostic pop

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <array>
#include <limits>
#include <set>

namespace py = pybind11;
using namespace TQMesh;

// ---------------------------------------------------------------------------
// Helper: auto-compute quadtree scale from coordinate extents.
// TQMesh requires scale slightly larger than the domain extent.
// ---------------------------------------------------------------------------
static double compute_quadtree_scale(
    const std::vector<std::array<double,2>>& ext_verts)
{
    if (ext_verts.empty()) return 1.0;
    double xmin = ext_verts[0][0], xmax = ext_verts[0][0];
    double ymin = ext_verts[0][1], ymax = ext_verts[0][1];
    for (const auto& v : ext_verts) {
        xmin = std::min(xmin, v[0]);
        xmax = std::max(xmax, v[0]);
        ymin = std::min(ymin, v[1]);
        ymax = std::max(ymax, v[1]);
    }
    // QuadTree is centered at (0,0), so scale must encompass all coordinates.
    // scale/2 must be ≥ max absolute coordinate value (with margin).
    double abs_max = std::max({std::abs(xmin), std::abs(xmax),
                               std::abs(ymin), std::abs(ymax)});
    // Also consider the extent to ensure proper cell coverage
    double extent = std::max(xmax - xmin, ymax - ymin);
    return std::max(abs_max * 2.0, extent) * 1.5;
}

static size_t split_long_boundary_edges(
    Boundary& boundary,
    Domain& domain,
    double max_length,
    size_t max_splits)
{
    if (max_length <= 0.0 || !std::isfinite(max_length) || max_splits == 0) {
        return 0;
    }

    std::vector<Edge*> work;
    work.reserve(boundary.edges().size());
    for (const auto& e_ptr : boundary.edges()) {
        if (e_ptr) {
            work.push_back(e_ptr.get());
        }
    }

    size_t splits = 0;
    while (!work.empty()) {
        Edge* edge = work.back();
        work.pop_back();
        if (!edge) {
            continue;
        }
        if (edge->length() <= max_length) {
            continue;
        }
        if (splits >= max_splits) {
            break;
        }

        auto split_edges = boundary.split_edge(*edge, domain.vertices(), 0.5);
        if (!split_edges.first || !split_edges.second) {
            continue;
        }

        ++splits;
        work.push_back(split_edges.first);
        work.push_back(split_edges.second);
    }

    return splits;
}

// ---------------------------------------------------------------------------
// Main mesh generation function
// ---------------------------------------------------------------------------
py::dict generate_triangular_mesh(
    // Exterior boundary: vertices in CCW order
    const std::vector<std::array<double,2>>& ext_verts,
    const std::vector<int>&                  ext_colors,
    // Optional interior holes: each list of vertices in CW order
    const std::vector<std::vector<std::array<double,2>>>& int_boundaries,
    const std::vector<std::vector<int>>&                  int_colors,
    // Constraint/refinement zones: list of {verts, size}
    const std::vector<std::vector<std::array<double,2>>>& constraint_verts,
    const std::vector<double>&                            constraint_sizes,
    // Fixed interior edges represented as polylines
    const std::vector<std::vector<std::array<double,2>>>& fixed_edges,
    // Global target element size
    double target_size,
    // Optional quad-layer controls: [x0, y0, x1, y1, n_layers, first_height, growth_rate]
    const std::vector<std::array<double,7>>&             quad_layers,
    // Optional tri-to-quad conversion after triangulation
    bool tri_to_quad,
    // Smoothing passes (0 = none)
    int n_smooth = 3,
    // Optional pre-triangulation boundary splitting length threshold (<=0 disables)
    double boundary_split_max_length = 0.0
)
{
    if (ext_verts.size() < 3)
        throw std::invalid_argument("Exterior boundary needs at least 3 vertices");
    if (ext_colors.size() != ext_verts.size())
        throw std::invalid_argument("ext_colors length must equal ext_verts length");

    // ---- Set quadtree scale (must happen before Domain construction) -------
    double scale = compute_quadtree_scale(ext_verts);
    TQMeshSetup::get_instance().set_quadtree_scale(scale);

    // ---- Size function: constant global size, overridden by constraints ----
    const double global_size = std::max(target_size, 1e-10);
    UserSizeFunction f = [global_size](const Vec2d&) { return global_size; };

    Domain domain { f };

    // ---- Exterior boundary -------------------------------------------------
    size_t boundary_splits_applied = 0;
    const double split_max_length =
        (std::isfinite(boundary_split_max_length) && boundary_split_max_length > 0.0)
            ? boundary_split_max_length
            : 0.0;
    const size_t split_limit = 200000;

    Boundary& b_ext = domain.add_exterior_boundary();
    {
        std::vector<Vec2d> coords;
        coords.reserve(ext_verts.size());
        for (const auto& v : ext_verts)
            coords.emplace_back(v[0], v[1]);

        std::vector<int> colors(ext_colors.begin(), ext_colors.end());
        b_ext.set_shape_from_coordinates(coords, colors);
        boundary_splits_applied += split_long_boundary_edges(
            b_ext,
            domain,
            split_max_length,
            split_limit - boundary_splits_applied);
    }

    // ---- Interior holes (clockwise) ----------------------------------------
    for (size_t i = 0; i < int_boundaries.size(); ++i) {
        const auto& iverts = int_boundaries[i];
        if (iverts.size() < 3) continue;

        std::vector<int> icolors;
        if (i < int_colors.size() && int_colors[i].size() == iverts.size()) {
            icolors.assign(int_colors[i].begin(), int_colors[i].end());
        } else {
            icolors.assign(iverts.size(), 1);
        }

        Boundary& b_int = domain.add_interior_boundary();
        std::vector<Vec2d> coords;
        coords.reserve(iverts.size());
        for (const auto& v : iverts)
            coords.emplace_back(v[0], v[1]);

        b_int.set_shape_from_coordinates(coords, icolors);
        boundary_splits_applied += split_long_boundary_edges(
            b_int,
            domain,
            split_max_length,
            split_limit - boundary_splits_applied);
    }

    // ---- Constraint/refinement zones as seeded vertices --------------------
    // TQMesh allows per-vertex local mesh size; we seed the constraint zone
    // boundary with vertices carrying the refined target size.
    for (size_t zi = 0; zi < constraint_verts.size(); ++zi) {
        const auto& zverts = constraint_verts[zi];
        double zsize = (zi < constraint_sizes.size()) ? constraint_sizes[zi] : global_size;
        double zrange = zsize * 4.0;   // influence radius ~4x element size

        for (const auto& v : zverts) {
            // add_vertex with (x, y, local_size, influence_range)
            domain.add_vertex(v[0], v[1], zsize, zrange);
        }
    }

    // ---- Fixed interior edges ----------------------------------------------
    struct CachedVertex {
        double x;
        double y;
        Vertex* v;
    };

    std::vector<CachedVertex> vertex_cache;
    vertex_cache.reserve(domain.vertices().size() + 256);
    for (const auto& vp : domain.vertices()) {
        if (!vp) {
            continue;
        }
        vertex_cache.push_back(CachedVertex{vp->xy().x, vp->xy().y, vp.get()});
    }

    const double vertex_merge_tol = std::max(global_size * 1.0e-8, 1.0e-9);
    auto get_or_add_vertex = [&](double x, double y) -> Vertex& {
        for (auto& cv : vertex_cache) {
            const double dx = x - cv.x;
            const double dy = y - cv.y;
            if (std::hypot(dx, dy) <= vertex_merge_tol) {
                return *(cv.v);
            }
        }
        Vertex& v_new = domain.add_vertex(
            x,
            y,
            global_size,
            std::max(global_size * 4.0, vertex_merge_tol * 4.0));
        vertex_cache.push_back(CachedVertex{x, y, &v_new});
        return v_new;
    };

    std::set<std::pair<const Vertex*, const Vertex*>> fixed_seen;
    size_t fixed_edges_added = 0;
    for (const auto& line : fixed_edges) {
        if (line.size() < 2) {
            continue;
        }
        Vertex* prev = nullptr;
        for (const auto& p : line) {
            Vertex& cur = get_or_add_vertex(p[0], p[1]);
            if (prev != nullptr && prev != &cur) {
                const Vertex* a = prev;
                const Vertex* b = &cur;
                if (b < a) {
                    std::swap(a, b);
                }
                const std::pair<const Vertex*, const Vertex*> key(a, b);
                if (fixed_seen.insert(key).second) {
                    domain.add_fixed_edge(*prev, cur);
                    ++fixed_edges_added;
                }
            }
            prev = &cur;
        }
    }

    // ---- Generate mesh -----------------------------------------------------
    MeshGenerator generator {};
    Mesh& mesh = generator.new_mesh(domain);

    // Distinguish bad boundary/domain input from downstream front evolution
    // failures, so diagnostics are explicit at the Python layer.
    if (!EntityChecks::check_domain_validity(domain)) {
        std::ostringstream oss;
        oss
            << "TQMesh domain validity failure"
            << "; ext_vertices=" << ext_verts.size()
            << "; n_holes=" << int_boundaries.size()
            << "; n_constraints=" << constraint_verts.size()
            << "; n_fixed_edges=" << fixed_edges.size()
            << "; n_fixed_edges_added=" << fixed_edges_added
            << "; target_size=" << global_size
            << "; n_quad_layers=" << quad_layers.size()
            << "; boundary_split_max_length=" << split_max_length
            << "; boundary_splits_applied=" << boundary_splits_applied;
        throw std::runtime_error(oss.str());
    }

    for (const auto& layer : quad_layers) {
        const int n_layers = std::max(0, static_cast<int>(std::lround(layer[4])));
        const double first_height = std::max(layer[5], 1.0e-10);
        const double growth_rate = std::max(layer[6], 1.0e-10);
        if (n_layers <= 0) {
            continue;
        }
        bool ok = generator.quad_layer_generation(mesh)
            .n_layers(static_cast<size_t>(n_layers))
            .first_height(first_height)
            .growth_rate(growth_rate)
            .starting_position(layer[0], layer[1])
            .ending_position(layer[2], layer[3])
            .generate_elements();
        if (!ok) {
            throw std::runtime_error("TQMesh: quad layer generation failed");
        }
    }

    // ---- Release the Python GIL during all heavy C++ computation -----------
    // This allows other Python threads (e.g. the QGIS bridge QTimer) to run
    // while TQMesh is generating elements and smoothing.
    {
        py::gil_scoped_release release_gil;

        generator.triangulation(mesh).generate_elements();

        if (tri_to_quad) {
            generator.tri2quad_modification(mesh).modify();
        }

        if (n_smooth > 0) {
            generator.mixed_smoothing(mesh).smooth(n_smooth);
        }

        MeshCleanup::assign_mesh_indices(mesh);
        MeshCleanup::setup_facet_connectivity(mesh);

        const size_t front_edges = mesh.get_front_edges().size();
        const bool validity_ok = EntityChecks::check_mesh_validity(mesh);
        size_t invalid_interior_edges = 0;
        size_t invalid_boundary_edges = 0;
        std::vector<std::string> invalid_boundary_edge_samples;
        std::vector<std::string> front_edge_samples;
        auto facet_count = [](const Vertex& v) {
            size_t cnt = 0;
            for (auto _f : v.facets()) {
                (void)_f;
                ++cnt;
            }
            return cnt;
        };
        if (!validity_ok) {
            auto edge_connected = [](const auto& e_ptr) {
                const Vertex& v1 = e_ptr->v1();
                const Vertex& v2 = e_ptr->v2();
                bool check_1 = false;
                bool check_2 = false;
                for (auto f : v1.facets()) {
                    check_1 = (f->get_edge_index(v1, v2) >= 0);
                    if (check_1) {
                        break;
                    }
                }
                for (auto f : v2.facets()) {
                    check_2 = (f->get_edge_index(v1, v2) >= 0);
                    if (check_2) {
                        break;
                    }
                }
                return check_1 && check_2;
            };

            for (const auto& e_ptr : mesh.interior_edges()) {
                if (!edge_connected(e_ptr)) {
                    ++invalid_interior_edges;
                }
            }
            for (const auto& e_ptr : mesh.boundary_edges()) {
                if (!edge_connected(e_ptr)) {
                    ++invalid_boundary_edges;
                    if (invalid_boundary_edge_samples.size() < 5) {
                        const Vertex& v1 = e_ptr->v1();
                        const Vertex& v2 = e_ptr->v2();
                        std::ostringstream eoss;
                        eoss
                            << "[(" << v1.xy().x << "," << v1.xy().y << ")->(" << v2.xy().x << "," << v2.xy().y << ")"
                            << ", f1=" << facet_count(v1)
                            << ", f2=" << facet_count(v2)
                            << ", color=" << e_ptr->color()
                            << "]";
                        invalid_boundary_edge_samples.push_back(eoss.str());
                    }
                }
            }
        }
        if (front_edges > 0) {
            for (const auto& e_ptr : mesh.get_front_edges()) {
                if (front_edge_samples.size() >= 5) {
                    break;
                }
                const Vertex& v1 = e_ptr->v1();
                const Vertex& v2 = e_ptr->v2();
                std::ostringstream foss;
                foss
                    << "[(" << v1.xy().x << "," << v1.xy().y << ")->(" << v2.xy().x << "," << v2.xy().y << ")"
                    << ", f1=" << facet_count(v1)
                    << ", f2=" << facet_count(v2)
                    << "]";
                front_edge_samples.push_back(foss.str());
            }
        }
        const double mesh_area = mesh.area();
        const double domain_area = domain.area();
        auto extent = domain.extent();
        const double scale_x = extent.first[1] - extent.first[0];
        const double scale_y = extent.second[1] - extent.second[0];
        const double area_diff = std::abs(mesh_area - domain_area);
        const double area_denom = std::max(std::abs(scale_x * scale_y), 1.0e-30);
        const double area_rel_diff = area_diff / area_denom;
        const bool area_ok = area_rel_diff < 1.0e-10;
        const bool complete = (front_edges == 0) && validity_ok && area_ok;
        if (!complete) {
            std::ostringstream oss;
            oss
                << "TQMesh completeness failure"
                << "; front_edges=" << front_edges
                << "; validity_ok=" << (validity_ok ? 1 : 0)
                << "; invalid_interior_edges=" << invalid_interior_edges
                << "; invalid_boundary_edges=" << invalid_boundary_edges
                << "; area_ok=" << (area_ok ? 1 : 0)
                << "; area_rel_diff=" << area_rel_diff
                << "; mesh_area=" << mesh_area
                << "; domain_area=" << domain_area
                << "; scale_x=" << scale_x
                << "; scale_y=" << scale_y
                << "; n_vertices=" << mesh.n_vertices()
                << "; n_triangles=" << mesh.n_triangles()
                << "; n_quads=" << mesh.n_quads()
                << "; target_size=" << global_size
                << "; n_smooth=" << n_smooth
                << "; tri_to_quad=" << (tri_to_quad ? 1 : 0)
                << "; boundary_split_max_length=" << split_max_length
                << "; boundary_splits_applied=" << boundary_splits_applied
                << "; n_holes=" << int_boundaries.size()
                << "; n_constraints=" << constraint_verts.size()
                << "; n_fixed_edges=" << fixed_edges.size()
                << "; n_fixed_edges_added=" << fixed_edges_added
                << "; n_quad_layers=" << quad_layers.size();
            if (!invalid_boundary_edge_samples.empty()) {
                oss << "; invalid_boundary_edge_samples=";
                for (size_t i = 0; i < invalid_boundary_edge_samples.size(); ++i) {
                    if (i) {
                        oss << ",";
                    }
                    oss << invalid_boundary_edge_samples[i];
                }
            }
            if (!front_edge_samples.empty()) {
                oss << "; front_edge_samples=";
                for (size_t i = 0; i < front_edge_samples.size(); ++i) {
                    if (i) {
                        oss << ",";
                    }
                    oss << front_edge_samples[i];
                }
            }
            throw std::runtime_error(oss.str());
        }
    }  // GIL re-acquired here

    // ---- Prepare output indices --------------------------------------------
    // TQMesh assigns indices via MeshCleanup (triggered by write_mesh or manually)
    MeshCleanup::assign_mesh_indices(mesh);
    MeshCleanup::setup_facet_connectivity(mesh);

    const size_t n_verts = mesh.n_vertices();
    const size_t n_tris  = mesh.n_triangles();
    const size_t n_quads = mesh.n_quads();

    // Collect vertices
    std::vector<double> out_x(n_verts), out_y(n_verts);
    for (const auto& vp : mesh.vertices()) {
        size_t idx = static_cast<size_t>(vp->index());
        out_x[idx] = vp->xy().x;
        out_y[idx] = vp->xy().y;
    }

    // Collect triangles
    std::vector<int32_t> out_tris(n_tris * 3);
    {
        size_t ti = 0;
        for (const auto& tp : mesh.triangles()) {
            out_tris[ti*3+0] = static_cast<int32_t>(tp->v1().index());
            out_tris[ti*3+1] = static_cast<int32_t>(tp->v2().index());
            out_tris[ti*3+2] = static_cast<int32_t>(tp->v3().index());
            ++ti;
        }
    }

    // Collect quads
    std::vector<int32_t> out_quads(n_quads * 4);
    {
        size_t qi = 0;
        for (const auto& qp : mesh.quads()) {
            out_quads[qi*4+0] = static_cast<int32_t>(qp->v1().index());
            out_quads[qi*4+1] = static_cast<int32_t>(qp->v2().index());
            out_quads[qi*4+2] = static_cast<int32_t>(qp->v3().index());
            out_quads[qi*4+3] = static_cast<int32_t>(qp->v4().index());
            ++qi;
        }
    }

    // Collect boundary edges
    auto bdry_edges = mesh.get_valid_boundary_edges();
    const size_t n_be = bdry_edges.size();
    std::vector<int32_t> out_bv0(n_be), out_bv1(n_be), out_bc(n_be);
    for (size_t i = 0; i < n_be; ++i) {
        out_bv0[i] = static_cast<int32_t>(bdry_edges[i]->v1().index());
        out_bv1[i] = static_cast<int32_t>(bdry_edges[i]->v2().index());
        out_bc[i]  = static_cast<int32_t>(bdry_edges[i]->color());
    }

    // ---- Pack into numpy arrays and return ---------------------------------
    py::dict result;

    result["verts_x"] = py::array_t<double>(
        {(py::ssize_t)n_verts}, out_x.data());
    result["verts_y"] = py::array_t<double>(
        {(py::ssize_t)n_verts}, out_y.data());
    result["triangles"] = py::array_t<int32_t>(
        {(py::ssize_t)n_tris, (py::ssize_t)3}, out_tris.data());
    result["quads"] = py::array_t<int32_t>(
        {(py::ssize_t)n_quads, (py::ssize_t)4}, out_quads.data());
    result["bdry_v0"]    = py::array_t<int32_t>({(py::ssize_t)n_be}, out_bv0.data());
    result["bdry_v1"]    = py::array_t<int32_t>({(py::ssize_t)n_be}, out_bv1.data());
    result["bdry_color"] = py::array_t<int32_t>({(py::ssize_t)n_be}, out_bc.data());
    result["n_fixed_edges_input"] = py::int_(fixed_edges.size());
    result["n_fixed_edges_added"] = py::int_(fixed_edges_added);

    return result;
}

// ---------------------------------------------------------------------------
PYBIND11_MODULE(hydra_tqmesh, m)
{
    m.doc() = "TQMesh advancing-front mesh generator binding for Backwater SWE2D";

    m.def("generate_triangular_mesh",
          &generate_triangular_mesh,
          py::arg("ext_verts"),
          py::arg("ext_colors"),
          py::arg("int_boundaries")   = std::vector<std::vector<std::array<double,2>>>{},
          py::arg("int_colors")       = std::vector<std::vector<int>>{},
          py::arg("constraint_verts") = std::vector<std::vector<std::array<double,2>>>{},
          py::arg("constraint_sizes") = std::vector<double>{},
          py::arg("fixed_edges")      = std::vector<std::vector<std::array<double,2>>>{},
          py::arg("target_size")      = 10.0,
          py::arg("quad_layers")      = std::vector<std::array<double,7>>{},
          py::arg("tri_to_quad")      = false,
          py::arg("n_smooth")         = 3,
          py::arg("boundary_split_max_length") = 0.0,
          R"doc(
Generate a triangular mesh using TQMesh's advancing-front algorithm.

Parameters
----------
ext_verts : list of (x, y) tuples
    Exterior boundary vertices in counter-clockwise order.
ext_colors : list of int
    Integer BC color for each exterior boundary edge (same length as ext_verts).
int_boundaries : list of list of (x, y)
    Optional interior hole boundaries in clockwise order.
int_colors : list of list of int
    BC colors per interior boundary edge.
constraint_verts : list of list of (x, y)
    Seed vertices for refinement zones (TQMesh local-size hints).
constraint_sizes : list of float
    Target element size for each constraint zone.
fixed_edges : list of list of (x, y)
    Interior constrained polylines. Each consecutive point pair is added
    as a fixed edge.
target_size : float
    Global target element size.
quad_layers : list of [x0, y0, x1, y1, n_layers, first_height, growth_rate]
    Optional boundary-aligned quad layer controls applied before triangulation.
tri_to_quad : bool
    If True, run TQMesh's triangle-to-quad modification after triangulation.
n_smooth : int
    Number of mixed-smoothing passes (default 3).
boundary_split_max_length : float
    Split boundary edges longer than this threshold before triangulation.
    Set <=0 to disable (default).

Returns
-------
dict with keys:
    verts_x, verts_y : float64 arrays (N_v,)
    triangles        : int32  array   (N_t, 3)
    quads            : int32  array   (N_q, 4)
    bdry_v0, bdry_v1 : int32 arrays  (N_be,)
    bdry_color       : int32  array   (N_be,)
)doc");
}
