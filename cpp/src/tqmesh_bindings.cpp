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
#include <cstring>

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
    // Optional quad-refinement passes after tri-to-quad
    int quad_refinements = 0,
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
        const int quad_refine_passes = std::max(0, quad_refinements);

        generator.triangulation(mesh).generate_elements();

        if (tri_to_quad) {
            generator.tri2quad_modification(mesh).modify();
        }

        for (int i = 0; i < quad_refine_passes; ++i) {
            if (!generator.quad_refinement(mesh).refine()) {
                throw std::runtime_error("TQMesh: quad refinement failed");
            }
            MeshCleanup::setup_facet_connectivity(mesh);
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
                << "; quad_refinements=" << quad_refine_passes
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
    py::array_t<double> verts_x_arr({(py::ssize_t)n_verts});
    py::array_t<double> verts_y_arr({(py::ssize_t)n_verts});
    py::array_t<int32_t> tris_arr({(py::ssize_t)n_tris, (py::ssize_t)3});
    py::array_t<int32_t> quads_arr({(py::ssize_t)n_quads, (py::ssize_t)4});
    py::array_t<int32_t> bdry_v0_arr({(py::ssize_t)n_be});
    py::array_t<int32_t> bdry_v1_arr({(py::ssize_t)n_be});
    py::array_t<int32_t> bdry_color_arr({(py::ssize_t)n_be});

    if (n_verts > 0) {
        std::memcpy(verts_x_arr.mutable_data(), out_x.data(), n_verts * sizeof(double));
        std::memcpy(verts_y_arr.mutable_data(), out_y.data(), n_verts * sizeof(double));
    }
    if (!out_tris.empty()) {
        std::memcpy(tris_arr.mutable_data(), out_tris.data(), out_tris.size() * sizeof(int32_t));
    }
    if (!out_quads.empty()) {
        std::memcpy(quads_arr.mutable_data(), out_quads.data(), out_quads.size() * sizeof(int32_t));
    }
    if (n_be > 0) {
        std::memcpy(bdry_v0_arr.mutable_data(), out_bv0.data(), n_be * sizeof(int32_t));
        std::memcpy(bdry_v1_arr.mutable_data(), out_bv1.data(), n_be * sizeof(int32_t));
        std::memcpy(bdry_color_arr.mutable_data(), out_bc.data(), n_be * sizeof(int32_t));
    }

    result["verts_x"] = std::move(verts_x_arr);
    result["verts_y"] = std::move(verts_y_arr);
    result["triangles"] = std::move(tris_arr);
    result["quads"] = std::move(quads_arr);
    result["bdry_v0"] = std::move(bdry_v0_arr);
    result["bdry_v1"] = std::move(bdry_v1_arr);
    result["bdry_color"] = std::move(bdry_color_arr);
    result["n_fixed_edges_input"] = py::int_(fixed_edges.size());
    result["n_fixed_edges_added"] = py::int_(fixed_edges_added);

    return result;
}

py::dict generate_merged_triangular_meshes(
    const py::list& mesh_specs,
    int receiver_index = 0,
    bool tri_to_quad = false,
    int n_smooth = 3,
    int quad_refinements = 0,
    double boundary_split_max_length = 0.0,
    int post_merge_smooth = 0
)
{
    if (mesh_specs.empty()) {
        throw std::invalid_argument("mesh_specs must contain at least one mesh specification");
    }

    struct MeshSpecData {
        std::vector<std::array<double, 2>> ext_verts;
        std::vector<int> ext_colors;
        std::vector<std::vector<std::array<double, 2>>> int_boundaries;
        std::vector<std::vector<int>> int_colors;
        std::vector<std::vector<std::array<double, 2>>> constraint_verts;
        std::vector<double> constraint_sizes;
        std::vector<std::vector<std::array<double, 2>>> fixed_edges;
        std::vector<std::array<double, 7>> quad_layers;
        double target_size = 10.0;
        bool tri_to_quad = false;
        int n_smooth = 0;
        int quad_refinements = 0;
        double boundary_split_max_length = 0.0;
        int mesh_id = 1;
        int element_color = 1;
    };

    std::vector<MeshSpecData> specs;
    specs.reserve(static_cast<size_t>(mesh_specs.size()));

    for (py::ssize_t i = 0; i < mesh_specs.size(); ++i) {
        py::dict spec = py::cast<py::dict>(mesh_specs[i]);

        MeshSpecData data;
        data.ext_verts = py::cast<std::vector<std::array<double, 2>>>(spec["ext_verts"]);
        data.ext_colors = py::cast<std::vector<int>>(spec["ext_colors"]);

        if (spec.contains("int_boundaries")) {
            data.int_boundaries = py::cast<std::vector<std::vector<std::array<double, 2>>>>(spec["int_boundaries"]);
        }
        if (spec.contains("int_colors")) {
            data.int_colors = py::cast<std::vector<std::vector<int>>>(spec["int_colors"]);
        }
        if (spec.contains("constraint_verts")) {
            data.constraint_verts = py::cast<std::vector<std::vector<std::array<double, 2>>>>(spec["constraint_verts"]);
        }
        if (spec.contains("constraint_sizes")) {
            data.constraint_sizes = py::cast<std::vector<double>>(spec["constraint_sizes"]);
        }
        if (spec.contains("fixed_edges")) {
            data.fixed_edges = py::cast<std::vector<std::vector<std::array<double, 2>>>>(spec["fixed_edges"]);
        }
        if (spec.contains("quad_layers")) {
            data.quad_layers = py::cast<std::vector<std::array<double, 7>>>(spec["quad_layers"]);
        }
        if (spec.contains("target_size")) {
            data.target_size = py::cast<double>(spec["target_size"]);
        }
        data.tri_to_quad = spec.contains("tri_to_quad") ? py::cast<bool>(spec["tri_to_quad"]) : tri_to_quad;
        data.n_smooth = spec.contains("n_smooth") ? py::cast<int>(spec["n_smooth"]) : n_smooth;
        data.quad_refinements = spec.contains("quad_refinements")
            ? py::cast<int>(spec["quad_refinements"])
            : quad_refinements;
        data.boundary_split_max_length = spec.contains("boundary_split_max_length")
            ? py::cast<double>(spec["boundary_split_max_length"])
            : boundary_split_max_length;
        data.mesh_id = spec.contains("mesh_id") ? py::cast<int>(spec["mesh_id"]) : static_cast<int>(i + 1);
        data.element_color = spec.contains("element_color") ? py::cast<int>(spec["element_color"]) : data.mesh_id;

        if (data.ext_verts.size() < 3) {
            throw std::invalid_argument("Each mesh spec requires at least 3 exterior vertices");
        }
        if (data.ext_colors.size() != data.ext_verts.size()) {
            throw std::invalid_argument("Each mesh spec requires ext_colors length to equal ext_verts length");
        }

        specs.push_back(std::move(data));
    }

    if (receiver_index < 0 || receiver_index >= static_cast<int>(specs.size())) {
        throw std::invalid_argument("receiver_index out of range");
    }

    MeshGenerator generator {};
    double merged_qtree_scale = 1.0;
    for (const auto& spec : specs) {
        merged_qtree_scale = std::max(merged_qtree_scale, compute_quadtree_scale(spec.ext_verts));
    }
    TQMeshSetup::get_instance().set_quadtree_scale(merged_qtree_scale);

    std::vector<std::unique_ptr<Domain>> domains;
    domains.reserve(specs.size());
    std::vector<Mesh*> meshes;
    meshes.reserve(specs.size());
    std::vector<size_t> fixed_edges_input(specs.size(), 0);
    std::vector<size_t> fixed_edges_added(specs.size(), 0);

    for (size_t si = 0; si < specs.size(); ++si) {
        const MeshSpecData& data = specs[si];
        const double global_size = std::max(data.target_size, 1.0e-10);

        UserSizeFunction f = [global_size](const Vec2d&) { return global_size; };
        domains.push_back(std::make_unique<Domain>(f));
        Domain& domain = *domains.back();

        const double split_max_length =
            (std::isfinite(data.boundary_split_max_length) && data.boundary_split_max_length > 0.0)
                ? data.boundary_split_max_length
                : 0.0;
        const size_t split_limit = 200000;
        size_t boundary_splits_applied = 0;

        Boundary& b_ext = domain.add_exterior_boundary();
        {
            std::vector<Vec2d> coords;
            coords.reserve(data.ext_verts.size());
            for (const auto& v : data.ext_verts) {
                coords.emplace_back(v[0], v[1]);
            }

            std::vector<int> colors(data.ext_colors.begin(), data.ext_colors.end());
            b_ext.set_shape_from_coordinates(coords, colors);
            boundary_splits_applied += split_long_boundary_edges(
                b_ext,
                domain,
                split_max_length,
                split_limit - boundary_splits_applied);
        }

        for (size_t i = 0; i < data.int_boundaries.size(); ++i) {
            const auto& iverts = data.int_boundaries[i];
            if (iverts.size() < 3) {
                continue;
            }

            std::vector<int> icolors;
            if (i < data.int_colors.size() && data.int_colors[i].size() == iverts.size()) {
                icolors.assign(data.int_colors[i].begin(), data.int_colors[i].end());
            } else {
                icolors.assign(iverts.size(), 1);
            }

            Boundary& b_int = domain.add_interior_boundary();
            std::vector<Vec2d> coords;
            coords.reserve(iverts.size());
            for (const auto& v : iverts) {
                coords.emplace_back(v[0], v[1]);
            }

            b_int.set_shape_from_coordinates(coords, icolors);
            boundary_splits_applied += split_long_boundary_edges(
                b_int,
                domain,
                split_max_length,
                split_limit - boundary_splits_applied);
        }

        for (size_t zi = 0; zi < data.constraint_verts.size(); ++zi) {
            const auto& zverts = data.constraint_verts[zi];
            double zsize = (zi < data.constraint_sizes.size()) ? data.constraint_sizes[zi] : global_size;
            double zrange = zsize * 4.0;

            for (const auto& v : zverts) {
                domain.add_vertex(v[0], v[1], zsize, zrange);
            }
        }

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
        size_t fixed_count_added = 0;
        for (const auto& line : data.fixed_edges) {
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
                        ++fixed_count_added;
                    }
                }
                prev = &cur;
            }
        }

        if (!EntityChecks::check_domain_validity(domain)) {
            std::ostringstream oss;
            oss << "TQMesh domain validity failure for mesh index " << si
                << "; ext_vertices=" << data.ext_verts.size()
                << "; n_holes=" << data.int_boundaries.size()
                << "; n_constraints=" << data.constraint_verts.size()
                << "; n_fixed_edges=" << data.fixed_edges.size()
                << "; n_fixed_edges_added=" << fixed_count_added
                << "; target_size=" << global_size
                << "; n_quad_layers=" << data.quad_layers.size()
                << "; boundary_split_max_length=" << split_max_length
                << "; boundary_splits_applied=" << boundary_splits_applied;
            throw std::runtime_error(oss.str());
        }

        Mesh& mesh = generator.new_mesh(domain, data.mesh_id, data.element_color);
        meshes.push_back(&mesh);
        fixed_edges_input[si] = data.fixed_edges.size();
        fixed_edges_added[si] = fixed_count_added;
    }

    for (size_t si = 0; si < specs.size(); ++si) {
        MeshSpecData& data = specs[si];
        Mesh& mesh = *meshes[si];
        Domain& domain = *domains[si];

        for (const auto& layer : data.quad_layers) {
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
                std::ostringstream oss;
                oss << "TQMesh: quad layer generation failed for mesh index " << si;
                throw std::runtime_error(oss.str());
            }
        }

        {
            py::gil_scoped_release release_gil;

            generator.triangulation(mesh).generate_elements();

            if (data.tri_to_quad) {
                generator.tri2quad_modification(mesh).modify();
            }

            const int refine_passes = std::max(0, data.quad_refinements);
            for (int i = 0; i < refine_passes; ++i) {
                if (!generator.quad_refinement(mesh).refine()) {
                    std::ostringstream oss;
                    oss << "TQMesh: quad refinement failed for mesh index " << si;
                    throw std::runtime_error(oss.str());
                }
                MeshCleanup::setup_facet_connectivity(mesh);
            }

            if (data.n_smooth > 0) {
                generator.mixed_smoothing(mesh).smooth(data.n_smooth);
            }

            MeshCleanup::assign_mesh_indices(mesh);
            MeshCleanup::setup_facet_connectivity(mesh);

            const size_t front_edges = mesh.get_front_edges().size();
            const bool validity_ok = EntityChecks::check_mesh_validity(mesh);
            const double mesh_area = mesh.area();
            const double domain_area = domain.area();
            auto extent = domain.extent();
            const double scale_x = extent.first[1] - extent.first[0];
            const double scale_y = extent.second[1] - extent.second[0];
            const double area_diff = std::abs(mesh_area - domain_area);
            const double area_denom = std::max(std::abs(scale_x * scale_y), 1.0e-30);
            const double area_rel_diff = area_diff / area_denom;
            const bool area_ok = area_rel_diff < 1.0e-10;

            if (front_edges > 0 || !validity_ok || !area_ok) {
                std::ostringstream oss;
                oss
                    << "TQMesh completeness failure for mesh index " << si
                    << "; front_edges=" << front_edges
                    << "; validity_ok=" << (validity_ok ? 1 : 0)
                    << "; area_ok=" << (area_ok ? 1 : 0)
                    << "; area_rel_diff=" << area_rel_diff
                    << "; mesh_area=" << mesh_area
                    << "; domain_area=" << domain_area
                    << "; n_vertices=" << mesh.n_vertices()
                    << "; n_triangles=" << mesh.n_triangles()
                    << "; n_quads=" << mesh.n_quads();
                oss << "; quad_refinements=" << std::max(0, data.quad_refinements);
                throw std::runtime_error(oss.str());
            }
        }
    }

    Mesh* receiver = meshes[static_cast<size_t>(receiver_index)];
    for (size_t si = 0; si < meshes.size(); ++si) {
        if (static_cast<int>(si) == receiver_index) {
            continue;
        }
        const size_t receiver_cells_before = receiver->n_triangles() + receiver->n_quads();
        const size_t donor_cells = meshes[si]->n_triangles() + meshes[si]->n_quads();

        if (!generator.merge_meshes(*receiver, *meshes[si])) {
            std::ostringstream oss;
            oss << "TQMesh native merge failed for donor index " << si
                << " into receiver index " << receiver_index;
            throw std::runtime_error(oss.str());
        }

        const size_t receiver_cells_after = receiver->n_triangles() + receiver->n_quads();
        if (donor_cells > 0 && receiver_cells_after < (receiver_cells_before + donor_cells)) {
            std::ostringstream oss;
            oss << "TQMesh native merge produced no donor transfer for donor index " << si
                << " into receiver index " << receiver_index
                << "; receiver_cells_before=" << receiver_cells_before
                << "; donor_cells=" << donor_cells
                << "; receiver_cells_after=" << receiver_cells_after;
            throw std::runtime_error(oss.str());
        }
    }

    if (post_merge_smooth > 0) {
        py::gil_scoped_release release_gil;
        generator.mixed_smoothing(*receiver).smooth(post_merge_smooth);
    }

    MeshCleanup::assign_mesh_indices(*receiver);
    MeshCleanup::setup_facet_connectivity(*receiver);

    const size_t n_verts = receiver->n_vertices();
    const size_t n_tris = receiver->n_triangles();
    const size_t n_quads = receiver->n_quads();

    std::vector<double> out_x(n_verts), out_y(n_verts);
    for (const auto& vp : receiver->vertices()) {
        size_t idx = static_cast<size_t>(vp->index());
        out_x[idx] = vp->xy().x;
        out_y[idx] = vp->xy().y;
    }

    std::vector<int32_t> out_tris(n_tris * 3);
    std::vector<int32_t> out_tri_colors(n_tris);
    {
        size_t ti = 0;
        for (const auto& tp : receiver->triangles()) {
            out_tris[ti * 3 + 0] = static_cast<int32_t>(tp->v1().index());
            out_tris[ti * 3 + 1] = static_cast<int32_t>(tp->v2().index());
            out_tris[ti * 3 + 2] = static_cast<int32_t>(tp->v3().index());
            out_tri_colors[ti] = static_cast<int32_t>(tp->color());
            ++ti;
        }
    }

    std::vector<int32_t> out_quads(n_quads * 4);
    std::vector<int32_t> out_quad_colors(n_quads);
    {
        size_t qi = 0;
        for (const auto& qp : receiver->quads()) {
            out_quads[qi * 4 + 0] = static_cast<int32_t>(qp->v1().index());
            out_quads[qi * 4 + 1] = static_cast<int32_t>(qp->v2().index());
            out_quads[qi * 4 + 2] = static_cast<int32_t>(qp->v3().index());
            out_quads[qi * 4 + 3] = static_cast<int32_t>(qp->v4().index());
            out_quad_colors[qi] = static_cast<int32_t>(qp->color());
            ++qi;
        }
    }

    auto bdry_edges = receiver->get_valid_boundary_edges();
    const size_t n_be = bdry_edges.size();
    std::vector<int32_t> out_bv0(n_be), out_bv1(n_be), out_bc(n_be);
    for (size_t i = 0; i < n_be; ++i) {
        out_bv0[i] = static_cast<int32_t>(bdry_edges[i]->v1().index());
        out_bv1[i] = static_cast<int32_t>(bdry_edges[i]->v2().index());
        out_bc[i] = static_cast<int32_t>(bdry_edges[i]->color());
    }

    size_t total_fixed_input = 0;
    size_t total_fixed_added = 0;
    for (size_t i = 0; i < fixed_edges_input.size(); ++i) {
        total_fixed_input += fixed_edges_input[i];
        total_fixed_added += fixed_edges_added[i];
    }

    py::dict result;
    py::array_t<double> verts_x_arr({(py::ssize_t)n_verts});
    py::array_t<double> verts_y_arr({(py::ssize_t)n_verts});
    py::array_t<int32_t> tris_arr({(py::ssize_t)n_tris, (py::ssize_t)3});
    py::array_t<int32_t> tri_colors_arr({(py::ssize_t)n_tris});
    py::array_t<int32_t> quads_arr({(py::ssize_t)n_quads, (py::ssize_t)4});
    py::array_t<int32_t> quad_colors_arr({(py::ssize_t)n_quads});
    py::array_t<int32_t> bdry_v0_arr({(py::ssize_t)n_be});
    py::array_t<int32_t> bdry_v1_arr({(py::ssize_t)n_be});
    py::array_t<int32_t> bdry_color_arr({(py::ssize_t)n_be});

    if (n_verts > 0) {
        std::memcpy(verts_x_arr.mutable_data(), out_x.data(), n_verts * sizeof(double));
        std::memcpy(verts_y_arr.mutable_data(), out_y.data(), n_verts * sizeof(double));
    }
    if (!out_tris.empty()) {
        std::memcpy(tris_arr.mutable_data(), out_tris.data(), out_tris.size() * sizeof(int32_t));
    }
    if (!out_tri_colors.empty()) {
        std::memcpy(tri_colors_arr.mutable_data(), out_tri_colors.data(), out_tri_colors.size() * sizeof(int32_t));
    }
    if (!out_quads.empty()) {
        std::memcpy(quads_arr.mutable_data(), out_quads.data(), out_quads.size() * sizeof(int32_t));
    }
    if (!out_quad_colors.empty()) {
        std::memcpy(quad_colors_arr.mutable_data(), out_quad_colors.data(), out_quad_colors.size() * sizeof(int32_t));
    }
    if (n_be > 0) {
        std::memcpy(bdry_v0_arr.mutable_data(), out_bv0.data(), n_be * sizeof(int32_t));
        std::memcpy(bdry_v1_arr.mutable_data(), out_bv1.data(), n_be * sizeof(int32_t));
        std::memcpy(bdry_color_arr.mutable_data(), out_bc.data(), n_be * sizeof(int32_t));
    }

    result["verts_x"] = std::move(verts_x_arr);
    result["verts_y"] = std::move(verts_y_arr);
    result["triangles"] = std::move(tris_arr);
    result["tri_colors"] = std::move(tri_colors_arr);
    result["quads"] = std::move(quads_arr);
    result["quad_colors"] = std::move(quad_colors_arr);
    result["bdry_v0"] = std::move(bdry_v0_arr);
    result["bdry_v1"] = std::move(bdry_v1_arr);
    result["bdry_color"] = std::move(bdry_color_arr);
    result["n_fixed_edges_input"] = py::int_(total_fixed_input);
    result["n_fixed_edges_added"] = py::int_(total_fixed_added);
    result["merged_mesh_count"] = py::int_(specs.size());
    result["receiver_index"] = py::int_(receiver_index);

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
          py::arg("quad_refinements") = 0,
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
quad_refinements : int
    Number of quad-refinement passes after triangulation/tri2quad.
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

    m.def("generate_merged_triangular_meshes",
          &generate_merged_triangular_meshes,
          py::arg("mesh_specs"),
          py::arg("receiver_index") = 0,
          py::arg("tri_to_quad") = false,
          py::arg("n_smooth") = 3,
          py::arg("quad_refinements") = 0,
          py::arg("boundary_split_max_length") = 0.0,
          py::arg("post_merge_smooth") = 0,
          R"doc(
Generate multiple TQMesh domains and merge them natively with MeshGenerator.merge_meshes.

Parameters
----------
mesh_specs : list[dict]
    Each dict supports the same keys as generate_triangular_mesh plus:
    mesh_id, element_color, tri_to_quad, n_smooth, boundary_split_max_length.
    Optional: quad_refinements.
    Required keys per dict: ext_verts, ext_colors, target_size.
receiver_index : int
    Index of the mesh to keep as merge receiver.
tri_to_quad : bool
    Global fallback tri_to_quad if not present in a mesh spec.
n_smooth : int
    Global fallback smoothing passes if not present in a mesh spec.
quad_refinements : int
    Global fallback quad-refinement passes if not present in a mesh spec.
boundary_split_max_length : float
    Global fallback boundary split threshold if not present in a mesh spec.
post_merge_smooth : int
    Optional smoothing passes after all merges complete.

Returns
-------
dict with keys:
    verts_x, verts_y : float64 arrays (N_v,)
    triangles        : int32  array   (N_t, 3)
    tri_colors       : int32  array   (N_t,)  (facet color / region id)
    quads            : int32  array   (N_q, 4)
    quad_colors      : int32  array   (N_q,)  (facet color / region id)
    bdry_v0, bdry_v1 : int32 arrays  (N_be,)
    bdry_color       : int32  array   (N_be,)
)doc");
}
