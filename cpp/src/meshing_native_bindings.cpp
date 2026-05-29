#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace py = pybind11;

namespace {

using Point = std::array<double, 2>;

inline double point_distance(const Point& a, const Point& b) {
    return std::hypot(a[0] - b[0], a[1] - b[1]);
}

inline double polyline_length(const std::vector<Point>& points) {
    if (points.size() < 2) {
        return 0.0;
    }
    double out = 0.0;
    for (size_t i = 1; i < points.size(); ++i) {
        out += point_distance(points[i - 1], points[i]);
    }
    return out;
}

inline std::tuple<double, double> point_to_segment_distance_s(
    double px,
    double py,
    double ax,
    double ay,
    double bx,
    double by
) {
    const double vx = bx - ax;
    const double vy = by - ay;
    const double wx = px - ax;
    const double wy = py - ay;
    const double vv = vx * vx + vy * vy;
    if (vv <= 1.0e-20) {
        return {std::hypot(px - ax, py - ay), 0.0};
    }
    const double t = std::max(0.0, std::min(1.0, (wx * vx + wy * vy) / vv));
    const double qx = ax + t * vx;
    const double qy = ay + t * vy;
    return {std::hypot(px - qx, py - qy), t};
}

inline std::tuple<double, double> polyline_distance_and_s(const std::vector<Point>& points, double x, double y) {
    if (points.size() < 2) {
        return {std::numeric_limits<double>::infinity(), 0.0};
    }
    double best_d = std::numeric_limits<double>::infinity();
    double best_s = 0.0;
    double acc = 0.0;
    for (size_t i = 0; i + 1 < points.size(); ++i) {
        const double ax = points[i][0];
        const double ay = points[i][1];
        const double bx = points[i + 1][0];
        const double by = points[i + 1][1];
        const double seg = std::hypot(bx - ax, by - ay);
        auto [d, t] = point_to_segment_distance_s(x, y, ax, ay, bx, by);
        const double s = acc + t * seg;
        if (d < best_d) {
            best_d = d;
            best_s = s;
        }
        acc += seg;
    }
    return {best_d, best_s};
}

inline Point interp_polyline_fraction(const std::vector<Point>& points, double frac) {
    if (points.empty()) {
        return {0.0, 0.0};
    }
    if (points.size() == 1) {
        return points.front();
    }
    const double total = polyline_length(points);
    if (total <= 1.0e-12) {
        return points.front();
    }
    const double f = std::max(0.0, std::min(1.0, frac));
    const double target = f * total;
    double acc = 0.0;
    for (size_t i = 0; i + 1 < points.size(); ++i) {
        const Point& a = points[i];
        const Point& b = points[i + 1];
        const double seg = point_distance(a, b);
        if (seg <= 1.0e-15) {
            continue;
        }
        if (acc + seg >= target) {
            const double t = (target - acc) / seg;
            return {
                a[0] + t * (b[0] - a[0]),
                a[1] + t * (b[1] - a[1]),
            };
        }
        acc += seg;
    }
    return points.back();
}

inline std::vector<Point> sample_open_polyline(const std::vector<Point>& points, double step) {
    std::vector<Point> pts = points;
    if (pts.size() < 2) {
        return pts;
    }
    const double h = std::max(step, 1.0e-9);
    std::vector<Point> out;
    out.reserve(pts.size());
    out.push_back(pts.front());
    for (size_t i = 1; i < pts.size(); ++i) {
        const Point& a = pts[i - 1];
        const Point& b = pts[i];
        const double seg = point_distance(a, b);
        const int ndiv = std::max(1, static_cast<int>(std::ceil(seg / h)));
        for (int j = 1; j <= ndiv; ++j) {
            const double t = static_cast<double>(j) / static_cast<double>(ndiv);
            const Point p{
                a[0] + t * (b[0] - a[0]),
                a[1] + t * (b[1] - a[1]),
            };
            if (point_distance(p, out.back()) <= 1.0e-12) {
                continue;
            }
            out.push_back(p);
        }
    }
    return out;
}

inline std::vector<Point> sample_closed_polyline(const std::vector<Point>& points, double step) {
    std::vector<Point> pts = points;
    if (pts.size() < 3) {
        return pts;
    }
    if (point_distance(pts.front(), pts.back()) <= 1.0e-12) {
        pts.pop_back();
    }
    if (pts.size() < 3) {
        return pts;
    }

    const double h = std::max(step, 1.0e-9);
    std::vector<Point> out;
    out.reserve(pts.size());
    out.push_back(pts.front());
    for (size_t i = 0; i < pts.size(); ++i) {
        const Point& a = pts[i];
        const Point& b = pts[(i + 1) % pts.size()];
        const double seg = point_distance(a, b);
        const int ndiv = std::max(1, static_cast<int>(std::ceil(seg / h)));
        for (int j = 1; j <= ndiv; ++j) {
            const double t = static_cast<double>(j) / static_cast<double>(ndiv);
            const Point p{
                a[0] + t * (b[0] - a[0]),
                a[1] + t * (b[1] - a[1]),
            };
            if (point_distance(p, out.back()) <= 1.0e-12) {
                continue;
            }
            out.push_back(p);
        }
    }

    if (out.size() >= 2 && point_distance(out.front(), out.back()) <= 1.0e-12) {
        out.pop_back();
    }
    return out;
}

inline std::vector<Point> downsample_polyline_samples(const std::vector<Point>& points, int max_points) {
    std::vector<Point> pts = points;
    const int max_pts = std::max(4, max_points);
    if (static_cast<int>(pts.size()) <= max_pts) {
        return pts;
    }
    const int stride = std::max(1, static_cast<int>(std::ceil(static_cast<double>(pts.size()) / static_cast<double>(max_pts))));
    std::vector<Point> out;
    out.reserve(static_cast<size_t>(max_pts + 1));
    for (size_t i = 0; i < pts.size(); i += static_cast<size_t>(stride)) {
        out.push_back(pts[i]);
    }
    if (point_distance(out.back(), pts.back()) > 0.0) {
        out.push_back(pts.back());
    }
    return out;
}

struct TrueRun {
    bool valid = false;
    int start = 0;
    int end = 0;
    int len = 0;
};

inline TrueRun longest_cyclic_true_run(const std::vector<uint8_t>& mask) {
    const int n = static_cast<int>(mask.size());
    if (n <= 0) {
        return {};
    }
    bool any_true = false;
    for (uint8_t v : mask) {
        if (v) {
            any_true = true;
            break;
        }
    }
    if (!any_true) {
        return {};
    }

    int best_len = 0;
    int best_end = -1;
    int cur = 0;
    for (int i = 0; i < 2 * n; ++i) {
        const bool v = mask[static_cast<size_t>(i % n)] != 0;
        if (v) {
            cur = std::min(n, cur + 1);
            if (cur > best_len) {
                best_len = cur;
                best_end = i;
            }
        } else {
            cur = 0;
        }
    }

    if (best_len <= 0 || best_end < 0) {
        return {};
    }
    const int start = (best_end - best_len + 1) % n;
    const int end = best_end % n;
    return {true, start, end, best_len};
}

inline std::pair<double, double> polyline_overlap_fractions_open_impl(
    const std::vector<Point>& poly_a,
    const std::vector<Point>& poly_b,
    double sample_step,
    double near_tol,
    int max_points
) {
    if (poly_a.size() < 2 || poly_b.size() < 2) {
        return {0.0, 0.0};
    }

    const double tol = std::max(near_tol, 0.0);

    double axmin = poly_a[0][0], axmax = poly_a[0][0], aymin = poly_a[0][1], aymax = poly_a[0][1];
    for (const auto& p : poly_a) {
        axmin = std::min(axmin, p[0]);
        axmax = std::max(axmax, p[0]);
        aymin = std::min(aymin, p[1]);
        aymax = std::max(aymax, p[1]);
    }

    double bxmin = poly_b[0][0], bxmax = poly_b[0][0], bymin = poly_b[0][1], bymax = poly_b[0][1];
    for (const auto& p : poly_b) {
        bxmin = std::min(bxmin, p[0]);
        bxmax = std::max(bxmax, p[0]);
        bymin = std::min(bymin, p[1]);
        bymax = std::max(bymax, p[1]);
    }

    if (axmax < bxmin - tol || bxmax < axmin - tol || aymax < bymin - tol || bymax < aymin - tol) {
        return {0.0, 0.0};
    }

    std::vector<Point> pa = sample_open_polyline(poly_a, sample_step);
    std::vector<Point> pb = sample_open_polyline(poly_b, sample_step);
    if (max_points > 0) {
        pa = downsample_polyline_samples(pa, max_points);
        pb = downsample_polyline_samples(pb, max_points);
    }
    if (pa.size() < 2 || pb.size() < 2) {
        return {0.0, 0.0};
    }

    int count_ab = 0;
    for (const auto& p : pa) {
        const auto [d, _s] = polyline_distance_and_s(pb, p[0], p[1]);
        if (d <= near_tol) {
            ++count_ab;
        }
    }

    int count_ba = 0;
    for (const auto& p : pb) {
        const auto [d, _s] = polyline_distance_and_s(pa, p[0], p[1]);
        if (d <= near_tol) {
            ++count_ba;
        }
    }

    const double ov_ab = static_cast<double>(count_ab) / static_cast<double>(std::max<size_t>(1, pa.size()));
    const double ov_ba = static_cast<double>(count_ba) / static_cast<double>(std::max<size_t>(1, pb.size()));
    return {ov_ab, ov_ba};
}

py::dict interface_overlap_metrics_closed_impl(
    const std::vector<Point>& ring_a,
    const std::vector<Point>& ring_b,
    double sample_step,
    double near_tol,
    int max_points
) {
    std::vector<Point> pa = sample_closed_polyline(ring_a, sample_step);
    std::vector<Point> pb = sample_closed_polyline(ring_b, sample_step);
    pa = downsample_polyline_samples(pa, max_points);
    pb = downsample_polyline_samples(pb, max_points);

    py::dict out;
    if (pa.size() < 2 || pb.size() < 2) {
        out["overlap_ab"] = 0.0;
        out["overlap_ba"] = 0.0;
        out["endpoint_delta_ab_max"] = std::numeric_limits<double>::infinity();
        out["endpoint_delta_ba_max"] = std::numeric_limits<double>::infinity();
        out["endpoint_delta_ab_mean"] = std::numeric_limits<double>::infinity();
        out["endpoint_delta_ba_mean"] = std::numeric_limits<double>::infinity();
        return out;
    }

    std::vector<Point> pb_open = pb;
    std::vector<Point> pa_open = pa;
    pb_open.push_back(pb.front());
    pa_open.push_back(pa.front());

    std::vector<double> d_ab;
    std::vector<double> d_ba;
    d_ab.reserve(pa.size());
    d_ba.reserve(pb.size());

    std::vector<uint8_t> near_ab;
    std::vector<uint8_t> near_ba;
    near_ab.reserve(pa.size());
    near_ba.reserve(pb.size());

    int n_ab = 0;
    for (const auto& p : pa) {
        const auto [d, _s] = polyline_distance_and_s(pb_open, p[0], p[1]);
        d_ab.push_back(d);
        const uint8_t hit = (d <= near_tol) ? 1 : 0;
        near_ab.push_back(hit);
        n_ab += static_cast<int>(hit);
    }

    int n_ba = 0;
    for (const auto& p : pb) {
        const auto [d, _s] = polyline_distance_and_s(pa_open, p[0], p[1]);
        d_ba.push_back(d);
        const uint8_t hit = (d <= near_tol) ? 1 : 0;
        near_ba.push_back(hit);
        n_ba += static_cast<int>(hit);
    }

    const double overlap_ab = static_cast<double>(n_ab) / static_cast<double>(std::max<size_t>(1, near_ab.size()));
    const double overlap_ba = static_cast<double>(n_ba) / static_cast<double>(std::max<size_t>(1, near_ba.size()));

    auto endpoint_deltas = [](const std::vector<Point>& samples,
                              const std::vector<uint8_t>& mask,
                              const std::vector<Point>& ref_open) {
        const TrueRun run = longest_cyclic_true_run(mask);
        if (!run.valid || samples.empty()) {
            return std::pair<double, double>{
                std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity(),
            };
        }
        const Point& p0 = samples[static_cast<size_t>(run.start)];
        const Point& p1 = samples[static_cast<size_t>(run.end)];
        const auto [d0, _s0] = polyline_distance_and_s(ref_open, p0[0], p0[1]);
        const auto [d1, _s1] = polyline_distance_and_s(ref_open, p1[0], p1[1]);
        return std::pair<double, double>{std::max(d0, d1), 0.5 * (d0 + d1)};
    };

    const auto [ep_ab_max, ep_ab_mean] = endpoint_deltas(pa, near_ab, pb_open);
    const auto [ep_ba_max, ep_ba_mean] = endpoint_deltas(pb, near_ba, pa_open);

    out["overlap_ab"] = overlap_ab;
    out["overlap_ba"] = overlap_ba;
    out["endpoint_delta_ab_max"] = ep_ab_max;
    out["endpoint_delta_ba_max"] = ep_ba_max;
    out["endpoint_delta_ab_mean"] = ep_ab_mean;
    out["endpoint_delta_ba_mean"] = ep_ba_mean;
    return out;
}

py::dict project_ring_to_chain_impl(
    const std::vector<Point>& ring_points,
    const std::vector<Point>& chain_points,
    double near_tol
) {
    py::dict out;
    std::vector<Point> ring_proj;
    ring_proj.reserve(ring_points.size());

    bool moved_any = false;
    int point_bbox_reject_count = 0;

    if (ring_points.size() < 3 || chain_points.size() < 2) {
        out["ring_proj"] = ring_points;
        out["moved_any"] = false;
        out["point_bbox_reject_count"] = 0;
        return out;
    }

    const double tol = std::max(near_tol, 1.0e-12);
    const double chain_len = polyline_length(chain_points);
    if (chain_len <= 1.0e-12) {
        out["ring_proj"] = ring_points;
        out["moved_any"] = false;
        out["point_bbox_reject_count"] = 0;
        return out;
    }

    double xmin = chain_points[0][0];
    double xmax = chain_points[0][0];
    double ymin = chain_points[0][1];
    double ymax = chain_points[0][1];
    for (const auto& p : chain_points) {
        xmin = std::min(xmin, p[0]);
        xmax = std::max(xmax, p[0]);
        ymin = std::min(ymin, p[1]);
        ymax = std::max(ymax, p[1]);
    }
    xmin -= tol;
    xmax += tol;
    ymin -= tol;
    ymax += tol;

    for (const auto& p : ring_points) {
        if (p[0] < xmin || p[0] > xmax || p[1] < ymin || p[1] > ymax) {
            ++point_bbox_reject_count;
            ring_proj.push_back(p);
            continue;
        }

        const auto [d, s] = polyline_distance_and_s(chain_points, p[0], p[1]);
        if (std::isfinite(d) && d <= tol) {
            const double frac = std::max(0.0, std::min(1.0, s / chain_len));
            const Point q = interp_polyline_fraction(chain_points, frac);
            ring_proj.push_back(q);
            if (point_distance(q, p) > 1.0e-10) {
                moved_any = true;
            }
        } else {
            ring_proj.push_back(p);
        }
    }

    out["ring_proj"] = ring_proj;
    out["moved_any"] = moved_any;
    out["point_bbox_reject_count"] = point_bbox_reject_count;
    return out;
}

}  // namespace

PYBIND11_MODULE(hydra_meshing_native, m) {
    m.doc() = "Native geometry kernels for SWE2D meshing prebuild phases";

    m.def(
        "polyline_overlap_fractions_open",
        [](const std::vector<Point>& poly_a,
           const std::vector<Point>& poly_b,
           double sample_step,
           double near_tol,
           int max_points) {
            return polyline_overlap_fractions_open_impl(
                poly_a,
                poly_b,
                sample_step,
                near_tol,
                max_points
            );
        },
        py::arg("poly_a"),
        py::arg("poly_b"),
        py::arg("sample_step"),
        py::arg("near_tol"),
        py::arg("max_points") = 0
    );

    m.def(
        "interface_overlap_metrics_closed",
        &interface_overlap_metrics_closed_impl,
        py::arg("ring_a"),
        py::arg("ring_b"),
        py::arg("sample_step"),
        py::arg("near_tol"),
        py::arg("max_points") = 1800
    );

    m.def(
        "project_ring_to_chain",
        &project_ring_to_chain_impl,
        py::arg("ring_points"),
        py::arg("chain_points"),
        py::arg("near_tol")
    );
}
