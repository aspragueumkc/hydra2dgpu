import ras_commander as rc
prj = rc.init_ras_project('hec_ras_project/test3.prj', ras_version='Ras.exe', load_results_summary=False)
print('project_folder:', prj.project_folder)
print('project_name:', prj.project_name)
print('is_initialized:', prj.initialized)
print('geom entries count:', len(prj.get_geom_entries()))
print('geom entries:', prj.get_geom_entries().to_dict('records'))
