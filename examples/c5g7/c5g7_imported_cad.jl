# Lewis, E. E., et al. "Benchmark specification for Deterministic 2-D/3-D MOX fuel 
# assembly transport calculations without spatial homogenization (C5G7 MOX)." 
# NEA/NSC 280 (2001): 2001.
using MOCNeutronTransport

# Geometry
# ---------------------------------------------------------------------------------------
gmsh.initialize()
gmsh.merge("c5g7.step")

# Physical groups (materials and labels)
# ---------------------------------------------------------------------------------------
gmsh.model.add_cad_entity_names_to_physical_groups(2)
# Easy to determine using gmsh.fltk.run()
mat_to_color = Dict{String, NTuple{4, Int32}}()
mat_to_color["MATERIAL_UO2"]             = (  0, 170,   0, 255)
mat_to_color["MATERIAL_GUIDE_TUBE"]      = (  0,   0,   0, 255)
mat_to_color["MATERIAL_FISSION_CHAMBER"] = (124, 124, 124, 255)
mat_to_color["MATERIAL_MOX-4.3"]         = (170, 255, 255, 255)
mat_to_color["MATERIAL_MOX-7.0"]         = (  0, 170, 255, 255)
mat_to_color["MATERIAL_MOX-8.7"]         = (  0,  85, 255, 255)
color_to_ent = gmsh.model.get_entities_by_color(2)
gmsh.model.add_materials_to_physical_groups_by_color(mat_to_color, color_to_ent, 2)

## Overlay grid
## ---------------------------------------------------------------------------------------
## Finemesh
#core_bb = lattice_bb = raytracing_module = pin_bb = AAB(pin_pitch, pin_pitch)
## A single pin cell
#cell = MPACTCoarseCell(pin_bb) 
## A single ray tracing module
#rt_module = MPACTRayTracingModule(raytracing_module, [cell.id;;])
## A single 1x1 lattice
#lattice = MPACTLattice(lattice_bb, [rt_module.id;;])
#core = MPACTCore2D(core_bb, [lattice.id;;])
#
#gmsh_overlay_rectangular_grid(bounding_box, grid_material, grid_nx, grid_ny)
#
## Visualize geometry prior to meshing
## gmsh.fltk.run()
#
## Use incomplete elements.
## Use quad remesh experimental option?
## Mesh
#gmsh.model.mesh.set_size(gmsh.model.get_entities(0), mesh_char_len)
#gmsh.model.mesh.generate(2)
#
## Optimize the mesh
#for n in 1:mesh_optimization_iters
#    gmsh.model.mesh.optimize("Laplace2D")
#end
#
## Visualize the mesh
#gmsh.fltk.run()
#
## Write the mesh to file 
#gmsh.write(mesh_filename)
#
## Finalize gmsh
#gmsh.finalize()
#
## Read the abaqus mesh into a 2D unstructured mesh data structure
#mesh = read_abaqus_2d(mesh_filename)
