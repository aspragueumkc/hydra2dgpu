#!/usr/bin/env python
from pathlib import Path
import json
p=Path('test3.json')
with p.open() as f:
    data=json.load(f)
out=Path('hec_ras_project')
out.mkdir(exist_ok=True)

# write project file (plain HEC-RAS .prj format)
prj=out/'test3.prj'
prj.write_text('\n'.join(['Proj Title=test3','Proj Description=Generated from test3.json','Proj Folder=', 'Geom File=01','Plan File=01'])+'\n')

# write minimal plan file test3.01
plan=out/'test3.01'
plan.write_text('Program Version=6.6\nPlan Title=GeneratedPlan\n')

# write geometry file test3.g01 in HEC-RAS plain format with cross sections
lines=[]
lines.append('Program Version=6.6')
lines.append('Geom File=01')
lines.append('River Reach=GeneratedRiver,MainReach')
for sec in data['sections']:
    rs=str(sec['river_station'])
    geom=sec['geometry']
    xs=[pt[0] for pt in geom]
    zs=[pt[1] for pt in geom]
    minx=min(xs)
    stations=[x-minx for x in xs]
    lines.append(f'Type RM Length L Ch R = GeneratedRiver,MainReach,{rs},0,0,0')
    count=len(stations)
    lines.append(f'#Sta/Elev={count}')
    values=[]
    for s,z in zip(stations,zs):
        values.append(f'{s:.3f}')
        values.append(f'{z:.3f}')
    per_line=10
    width=8
    for i in range(0,len(values),per_line):
        block=values[i:i+per_line]
        line=''.join(v.rjust(width) for v in block)
        lines.append(line)

gfile=out/'test3.g01'
with gfile.open('w') as f:
    f.write('\n'.join(lines)+'\n')

print('Wrote files in', out)

# Try to initialize via ras_commander if available in this interpreter
try:
    import ras_commander as rc
    prj_obj = rc.init_ras_project(str(prj), ras_version='Ras.exe', load_results_summary=False)
    print('ras_commander initialized project:', prj_obj.is_initialized())
    geom_entries = prj_obj.get_geom_entries()
    print('geom entries count:', len(geom_entries))
except Exception as e:
    print('ras_commander not available or init failed:', e)
