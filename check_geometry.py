import backwater2 as bw

m = bw.load_input('example_project/example.gpkg')
for xs in m.sections:
    try:
        rs = float(xs.river_station)
    except:
        continue
    if abs(rs - 89.3) < 0.5:
        print(f'RS {xs.river_station}: has_culvert={xs.has_culvert()}')
        z_crown = xs.culvert_upstream_invert + xs.culvert_full_depth()
        print(f'  z_inlet={xs.culvert_upstream_invert}, z_crown={z_crown}')
        print(f'  geometry points: {len(xs.geometry)}')
        if xs.geometry:
            print('  First 15 points (x, z):')
            for i, (x, z) in enumerate(xs.geometry[:15]):
                above = 'YES' if z >= z_crown else 'no'
                print(f'    {i:2d}: x={x:7.2f}, z={z:7.2f}  above_crown={above}')
            print('  Last 5 points:')
            for i, (x, z) in enumerate(xs.geometry[-5:]):
                idx = len(xs.geometry) - 5 + i
                above = 'YES' if z >= z_crown else 'no'
                print(f'    {idx:2d}: x={x:7.2f}, z={z:7.2f}  above_crown={above}')
        break
