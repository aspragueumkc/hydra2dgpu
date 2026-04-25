import json
from pathlib import Path

import backwater2 as bw
from hec2.validation import HEC2InputParser
from hec2.standard_backwater import StandardBackwaterSolver
from hec2.profile_solver import CrossSectionData


def parse_rs(value):
    try:
        return float(value)
    except Exception:
        text = str(value)
        num = ""
        for ch in text:
            if ch.isdigit() or ch in ".-+eE":
                num += ch
        return float(num) if num else 0.0


def f6(v):
    return f"{float(v):6.1f}"[:6]


def f8(v):
    return f"{float(v):8.3f}"[:8]


def card(card_type, field6=0.0, fields8=None):
    if fields8 is None:
        fields8 = [0.0] * 9
    if len(fields8) < 9:
        fields8 = list(fields8) + [0.0] * (9 - len(fields8))
    return f"{card_type}{f6(field6)}" + "".join(f8(v) for v in fields8[:9])


def build_dat_from_json(json_path: Path, dat_path: Path):
    data = json.loads(json_path.read_text())
    sections = sorted(data["sections"], key=lambda s: parse_rs(s.get("river_station", 0.0)))

    xs_ds = bw.CrossSection(
        river_station=str(sections[0]["river_station"]),
        geometry=[(float(x), float(z)) for x, z in sections[0]["geometry"]],
        left_bank_station=float(sections[0]["left_bank_station"]),
        right_bank_station=float(sections[0]["right_bank_station"]),
        n_lob=float(sections[0]["n_lob"]),
        n_ch=float(sections[0]["n_ch"]),
        n_rob=float(sections[0]["n_rob"]),
    )

    q = float(data["flow_cfs"])
    bc_type = str(data["boundary_condition"])
    bc_value = float(data["boundary_value"])
    if bc_type == "normal_depth":
        downstream_wsel = float(bw.solve_normal_depth(xs_ds, q, bc_value))
    else:
        downstream_wsel = bc_value

    n_default = float(sections[0]["n_ch"])

    lines = []
    lines.append("T1  GENERATED FROM geopackage_export_test.json")

    # J1 uses field8(7)=Q, field8(8)=downstream WSEL, field8(9)=FQ
    j1_fields = [0.0] * 9
    j1_fields[6] = q
    j1_fields[7] = downstream_wsel
    j1_fields[8] = 1.0
    lines.append(card("J1", 0.0, j1_fields))

    # Uniform Manning n
    lines.append(card("NC", n_default, [0.0] * 9))

    # X1 + GR cards per cross section
    for i, sec in enumerate(sections):
        rs = float(parse_rs(sec.get("river_station", i)))
        pts = [(float(x), float(z)) for x, z in sec["geometry"]]
        npts = len(pts)

        # Parser expects this section's reach_length to be distance from previous section.
        reach_from_prev = 0.0
        if i > 0:
            prev = sections[i - 1]
            reach_from_prev = float(prev.get("L_ch_to_next", 0.0))

        x1_fields = [0.0] * 9
        x1_fields[0] = float(npts)
        x1_fields[4] = reach_from_prev
        lines.append(card("X1", rs, x1_fields))

        # GR: field6=elev1, field8[0]=sta1, then (elev,sta) pairs in field8[1:]
        chunk_size = 5
        for j in range(0, npts, chunk_size):
            chunk = pts[j:j + chunk_size]
            elev1 = chunk[0][1]
            sta1 = chunk[0][0]
            fields = [0.0] * 9
            fields[0] = sta1
            out_idx = 1
            for sta, elev in chunk[1:]:
                if out_idx + 1 >= 9:
                    break
                fields[out_idx] = elev
                fields[out_idx + 1] = sta
                out_idx += 2
            lines.append(card("GR", elev1, fields))

    lines.append("ER")
    dat_path.write_text("\n".join(lines) + "\n")
    return sections, downstream_wsel


def run_native(dat_path: Path):
    parser = HEC2InputParser(str(dat_path))
    if not parser.runs:
        raise RuntimeError("No runs parsed from generated DAT file")

    run = parser.runs[0]

    # Build CrossSectionData equivalent to runner._build_cross_section_data
    ordered = list(run.cross_sections)  # HEC-2 order is DS -> US
    cross_sections = []
    for xs in ordered:
        coords = sorted(xs.points, key=lambda p: p[0])
        cross_sections.append(
            CrossSectionData(
                station_id=str(xs.section_id),
                coordinates=coords,
                manning_n=xs.manning_n,
                distance_to_next=0.0,
            )
        )

    for i in range(len(cross_sections) - 1):
        next_reach_len = ordered[i + 1].reach_length
        if next_reach_len > 0:
            cross_sections[i].distance_to_next = next_reach_len
        else:
            try:
                curr_id = float(cross_sections[i].station_id)
                nxt_id = float(cross_sections[i + 1].station_id)
                dist = abs(nxt_id - curr_id)
                cross_sections[i].distance_to_next = dist if 0 < dist < 10000 else 100.0
            except Exception:
                cross_sections[i].distance_to_next = 100.0

    ds_min_elev = min(e for _, e in cross_sections[0].coordinates)
    if run.downstream_wsel is not None:
        downstream_depth = max(0.01, run.downstream_wsel - ds_min_elev)
    else:
        downstream_depth = 0.5

    solver = StandardBackwaterSolver()
    profile = solver.solve_profile_from_data(
        cross_sections=cross_sections,
        discharge=run.discharge,
        downstream_depth=downstream_depth,
    )
    return run, profile


def main():
    root = Path(__file__).resolve().parent
    json_path = root / "geopackage_export_test.json"
    dat_path = root / "geopackage_export_test_native.dat"

    sections, ds_wsel = build_dat_from_json(json_path, dat_path)
    run, profile = run_native(dat_path)

    print(f"DAT file: {dat_path}")
    print(f"Section order used: {[str(s['river_station']) for s in sections]}")
    print(f"Downstream WSEL used in J1: {ds_wsel:.6f} ft")

    print(f"Run: Q={run.discharge}")
    if not profile:
        print("  no profile results")
        return
    for sid, wsel, depth in profile:
        print(f"  station_id={sid} wsel={wsel:.6f} depth={depth:.6f}")


if __name__ == "__main__":
    main()
