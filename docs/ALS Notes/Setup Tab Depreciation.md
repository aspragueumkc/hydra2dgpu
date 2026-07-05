# HYDRA2D Model Setup Panel, Setup Tab Redesign

## The setup tab contains a 3 page toolbox:
- Load layers: Contains map layer selection widgets. Only page with clear scope that will remain mostly in its current state under new Setup tab redesign 
- Mesh Setup: This page really contains import export functions nothing to do with mesh setup. All methods accessed here will be relocated.
- Utilities: Only contains 2 widgets both accessible from the HYDRA2DGPU main menu. The page will be deleted.

## Changes to each page described here
### Load layers:
 - Delete node layer and cell layer widgets.
 - Remove "Create 2D Model Geopackage"  button widget.
 - Move "Autopopulate from Group" and "Refresh Layers" button widgets to top of Load layers page.
 - Rename page to "Layers"


Load 2D Model Geopackage
Save Mesh to Geopackage
Load Mesh From Selected Layers
Assign Mesh Node Z From Terrain
Pull Mesh Node Z From Nodes Layers
Export Results to Ugrid
Load Mesh from Geopackage



Import / Export
Export Mesh to Map Layers
Export Mesh to Ugrid
Load Mesh from Map Layers
 -Dialog to select node and cell layers
 -remove node and cell layesr from load layesr


Layers group Delete
Autopopulate from group Delete
Refresh layers Delete, make this function automatic when sele