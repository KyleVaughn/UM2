# Lewis, E. E., et al. "Benchmark specification for Deterministic 2-D/3-D MOX fuel 
# assembly transport calculations without spatial homogenization (C5G7 MOX)." 
# NEA/NSC 280 (2001): 2001.
using MOCNeutronTransport
filename = "c5g7.step"
add_timestamps_to_log()

# Import model and assign materials 
# ---------------------------------------------------------------------------------------
materials = import_model(filename, names = true)

# Construct and overlay MPACT grid hierarchy
# ---------------------------------------------------------------------------------------
push!(materials, Material("Moderator", (168, 50, 50, 255), 1.0))
boundingbox = AABox(64.26, 64.26)

# Lattices
lattice_div = [21.42, 2*21.42]
lattice_grid = RectilinearGrid(boundingbox, lattice_div, lattice_div)

# RT modules
module_grid = lattice_grid # assembly modular ray tracing

# Coarse grid
coarse_div = [1.26*i for i ∈ 1:17*3-1]
coarse_grid = RectilinearGrid(boundingbox, coarse_div, coarse_div) 

mpact_grid = MPACTGridHierarchy(lattice_grid, module_grid, coarse_grid)
# Overlay grid
# Use the material at the end of `materials` to fill empty space (the grid)
overlay_mpact_grid_hierarchy(mpact_grid, materials)

# Mesh
# ---------------------------------------------------------------------------------------
## Set mesh size by materials
#mat_to_mesh_size = Dict{String, Float32}()
#mat_to_mesh_size["MATERIAL_UO2"]             = 0.1 
#mat_to_mesh_size["MATERIAL_GUIDE_TUBE"]      = 0.1
#mat_to_mesh_size["MATERIAL_FISSION_CHAMBER"] = 0.1
#mat_to_mesh_size["MATERIAL_MOX-4.3"]         = 0.1
#mat_to_mesh_size["MATERIAL_MOX-7.0"]         = 0.1
#mat_to_mesh_size["MATERIAL_MOX-8.7"]         = 0.1

## Get material areas
#gmsh.model.mesh.set_size_by_material(mat_to_mesh_size)
#gmsh.model.mesh.generate(2)
## Optimize the mesh
#for n in 1:mesh_optimization_iters
#    gmsh.model.mesh.optimize("Laplace2D")
#end
## Write the mesh to file 
#gmsh.write(filename*"inp")
## Finalize gmsh
#gmsh.finalize()
#
## Convert to XDMF
#mesh = import_mesh(filename*"inp")
## Compute mesh area errors
#
#mesh_partition = partition_mesh(mesh)
#export_mesh(mesh_partition, filename*".xdmf")
