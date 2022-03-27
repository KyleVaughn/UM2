# Lewis, E. E., et al. "Benchmark specification for Deterministic 2-D/3-D MOX fuel 
# assembly transport calculations without spatial homogenization (C5G7 MOX)." 
# NEA/NSC 280 (2001): 2001.
using MOCNeutronTransport
using Statistics

filename = "c5g7.step"
add_timestamps_to_logger()

# Gmsh options (Will hide from use later)
gmsh.initialize()
gmsh.option.set_number("General.NumThreads", 0) # 0 uses system default, i.e. OMP_NUM_THREADS)
gmsh.option.set_number("Geometry.OCCParallel", 1) # Use parallel OCC boolean operations
#gmsh.option.set_number("General.Verbosity", 2) # Supress gmsh info
gmsh.option.set_number("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.set_number("Mesh.MeshSizeFromPoints", 0)
#gmsh.option.set_number("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.set_number("Mesh.Algorithm", 5) # Delaunay handles large element size gradients better

# Import model and assign materials 
# ---------------------------------------------------------------------------------------
materials = import_model(filename, names = true)

# Construct and overlay MPACT grid hierarchy
# ---------------------------------------------------------------------------------------
boundingbox = AABox(64.26, 64.26)

lattice_div = [21.42, 2*21.42]
lattice_grid = RectilinearGrid(boundingbox, lattice_div, lattice_div)

module_grid = lattice_grid # assembly modular ray tracing

coarse_div = [1.26*i for i ∈ 1:17*3-1]
coarse_grid = RectilinearGrid(boundingbox, coarse_div, coarse_div) 

mpact_grid = MPACTGridHierarchy(lattice_grid, module_grid, coarse_grid)

# Overlay grid and use the material at the end of `materials` to fill empty space (the grid)
push!(materials, Material(name="Moderator", color="sienna", mesh_size=1.0))
overlay_mpact_grid_hierarchy(mpact_grid, materials)

# Mesh
# ---------------------------------------------------------------------------------------
# Using 1 / 2Σₜ for mesh size, with Σₜ taken as the average Σₜᵢ in Table 1
# THIS IS NOT A PROPER 1-GROUP COLLAPSE
Σₜᵢ_FC  = [1.90730E-01, 4.56520E-01, 6.40700E-01, 6.49840E-01, 6.70630E-01, 8.75060E-01, 1.43450E+00]
Σₜᵢ_GT  = [1.90730E-01, 4.56520E-01, 6.40670E-01, 6.49670E-01, 6.70580E-01, 8.75050E-01, 1.43450E+00]
Σₜᵢ_M43 = [2.11920E-01, 3.55810E-01, 4.88900E-01, 5.71940E-01, 4.32390E-01, 6.84950E-01, 6.88910E-01]
Σₜᵢ_M70 = [2.14540E-01, 3.59350E-01, 4.98910E-01, 5.96220E-01, 4.80350E-01, 8.39360E-01, 8.59480E-01]
Σₜᵢ_M87 = [2.16280E-01, 3.61700E-01, 5.05630E-01, 6.11170E-01, 5.08900E-01, 9.26670E-01, 9.60990E-01]
Σₜᵢ_UO2 = [2.12450E-01, 3.55470E-01, 4.85540E-01, 5.59400E-01, 3.18030E-01, 4.01460E-01, 5.70610E-01]
Σₜᵢ_Mod = [2.30070E-01, 7.76460E-01, 1.48420E+00, 1.50520E+00, 1.55920E+00, 2.02540E+00, 3.30570E+00]

Σₜ_FC  = mean(Σₜᵢ_FC)  
Σₜ_GT  = mean(Σₜᵢ_GT)
Σₜ_M43 = mean(Σₜᵢ_M43)
Σₜ_M70 = mean(Σₜᵢ_M70)
Σₜ_M87 = mean(Σₜᵢ_M87)
Σₜ_UO2 = mean(Σₜᵢ_UO2)
Σₜ_Mod = mean(Σₜᵢ_Mod)

materials[1].mesh_size = 1 / 2Σₜ_FC  # Fission Chamber
materials[2].mesh_size = 1 / 2Σₜ_GT  # Guide Tube
materials[3].mesh_size = 1 / 2Σₜ_M43 # MOX-4.3% enriched
materials[4].mesh_size = 1 / 2Σₜ_M70 # MOX-7.0% enriched
materials[5].mesh_size = 1 / 2Σₜ_M87 # MOX-8.7% enriched
materials[6].mesh_size = 1 / 2Σₜ_UO2 # Uranium Oxide
materials[7].mesh_size = 1 / 2Σₜ_Mod # Moderator

set_mesh_field_using_materials(materials)
gmsh.model.mesh.generate(2)

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
