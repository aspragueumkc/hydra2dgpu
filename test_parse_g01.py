from ras_commander import geom
from pathlib import Path
p=Path('hec_ras_project')/'test3.g01'
print('file exists', p.exists())
try:
    df=geom.GeomCrossSection.get_station_elevation(str(p),'GeneratedRiver','MainReach','0')
    print('points', len(df))
    print(df.head().to_csv(index=False))
except Exception as e:
    print('error', e)
