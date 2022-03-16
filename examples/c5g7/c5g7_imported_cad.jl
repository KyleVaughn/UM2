# Lewis, E. E., et al. "Benchmark specification for Deterministic 2-D/3-D MOX fuel 
# assembly transport calculations without spatial homogenization (C5G7 MOX)." 
# NEA/NSC 280 (2001): 2001.
using MOCNeutronTransport

# Import geometry
# ---------------------------------------------------------------------------------------
gmsh.initialize()
gmsh.merge("c5g7.step")

# Add physical groups (materials and labels)
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

# Construct overlay and overlay MPACT grid hierarchy
# ---------------------------------------------------------------------------------------
# Lattices
boundingbox = AABox(64.26, 64.26)
lattice_div = [21.42, 2*21.42]
lattice_grid = RectilinearGrid(boundingbox, lattice_div, lattice_div)
# RT modules
module_grid = lattice_grid # assembly modular ray tracing
# Coarse grid
coarse_div = [1.26*i for i ∈ 1:17*3]
coarse_grid = RectilinearGrid(boundingbox, coarse_div, coarse_div) 

mpact_grid = MPACTGridHierarchy(lattice_grid, module_grid, coarse_grid)


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
