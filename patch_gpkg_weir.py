"""
Patch example_project/example.gpkg to add culvert weir fields and set RS 89.3
values matching the HEC-RAS calibration:
  culvert_weir_coeff   = 2.6
  culvert_weir_sta_left  = 0.0
  culvert_weir_sta_right = 33.67
"""
import sys, geopandas as gpd, fiona

gpkg = 'example_project/example.gpkg'
layer = 'cross_sections'

gdf = gpd.read_file(gpkg, layer=layer)
print('Columns before:', list(gdf.columns))

# Add weir fields if not present
for col, default in [('culvert_weir_coeff', 3.0),
                     ('culvert_weir_sta_left', 0.0),
                     ('culvert_weir_sta_right', 0.0)]:
    if col not in gdf.columns:
        gdf[col] = default
        print(f'  Added column {col} = {default}')

# Set RS 89.3 calibrated values
mask = gdf['river_station'].astype(str) == '89.3'
if mask.sum() == 0:
    print('WARNING: RS 89.3 not found; stations:', gdf['river_station'].tolist())
else:
    gdf.loc[mask, 'culvert_weir_coeff']    = 2.6
    gdf.loc[mask, 'culvert_weir_sta_left']  = 0.0
    gdf.loc[mask, 'culvert_weir_sta_right'] = 33.67
    print('  Set RS 89.3: Cw=2.6, sta_left=0, sta_right=33.67')

# Write back (overwrite cross_sections layer)
gdf.to_file(gpkg, layer=layer, driver='GPKG')
print('Done. Columns after:', [c for c in gdf.columns if c != 'geometry'])
