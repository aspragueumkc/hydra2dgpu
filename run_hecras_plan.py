from pathlib import Path
import ras_commander as rc

# Config
project_prj = Path('hec_ras_project') / 'test3.prj'
ras_exe = r"C:\Program Files (x86)\HEC\HEC-RAS\6.3.1\Ras.exe"
plan_number = '01'

print('Initializing project', project_prj)
prj = rc.init_ras_project(str(project_prj), ras_version=ras_exe, load_results_summary=False)
print('Project initialized:', getattr(prj, 'initialized', None))
print('Using RAS exe:', getattr(prj, 'ras_exe_path', None))

# Execute plan (may fail if HEC-RAS install or plan contents are incomplete)
try:
	success = rc.RasCmdr.compute_plan(plan_number, ras_object=prj, verify=True)
	print('Compute success:', success)
except Exception as e:
	print('Compute failed:', e)
