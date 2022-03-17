# Lewis, E. E., et al. "Benchmark specification for Deterministic 2-D/3-D MOX fuel 
# assembly transport calculations without spatial homogenization (C5G7 MOX)." 
# NEA/NSC 280 (2001): 2001.
using MOCNeutronTransport
filename = "c5g7"

# Import geometry
# ---------------------------------------------------------------------------------------
gmsh.initialize()
gmsh.merge(filename*".step")

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

# Construct and overlay MPACT grid hierarchy
# ---------------------------------------------------------------------------------------
boundingbox = AABox(64.26, 64.26)
# Lattices
lattice_div = [21.42, 2*21.42]
lattice_grid = RectilinearGrid(boundingbox, lattice_div, lattice_div)
# RT modules
module_grid = lattice_grid # assembly modular ray tracing
# Coarse grid
coarse_div = [1.26*i for i ∈ 1:17*3-1]
coarse_grid = RectilinearGrid(boundingbox, coarse_div, coarse_div) 
# Overlay grid
mpact_grid = MPACTGridHierarchy(lattice_grid, module_grid, coarse_grid)
mpact_grid = MPACTGridHierarchy(coarse_grid)
gmsh.model.overlay_mpact_grid_hierarchy(mpact_grid, "MATERIAL_WATER")
# Get material areas


# Mesh
# ---------------------------------------------------------------------------------------
# Set mesh size by materials
mat_to_mesh_size = Dict{String, Float32}()
mat_to_mesh_size["MATERIAL_UO2"]             = 0.1 
mat_to_mesh_size["MATERIAL_GUIDE_TUBE"]      = 0.1
mat_to_mesh_size["MATERIAL_FISSION_CHAMBER"] = 0.1
mat_to_mesh_size["MATERIAL_MOX-4.3"]         = 0.1
mat_to_mesh_size["MATERIAL_MOX-7.0"]         = 0.1
mat_to_mesh_size["MATERIAL_MOX-8.7"]         = 0.1
gmsh.model.mesh.set_size_by_material(mat_to_mesh_size)
gmsh.model.mesh.generate(2)
# Optimize the mesh
for n in 1:mesh_optimization_iters
    gmsh.model.mesh.optimize("Laplace2D")
end
# Write the mesh to file 
gmsh.write(filename*"inp")
# Finalize gmsh
gmsh.finalize()

# Convert to XDMF
mesh = import_mesh(filename*"inp")
# Compute mesh area errors

mesh_partition = partition_mesh(mesh)
export_mesh(mesh_partition, filename*".xdmf")
