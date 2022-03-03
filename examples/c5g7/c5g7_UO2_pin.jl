# Lewis, E. E., et al. "Benchmark specification for Deterministic 2-D/3-D MOX fuel 
# assembly transport calculations without spatial homogenization (C5G7 MOX)." 
# NEA/NSC 280 (2001): 2001.

using MOCNeutronTransport

const AAB = AABox2D{Float64}
const pin_pitch = 1.26 # pg 3
const pin_radius = 0.54 # pg 3
mesh_char_len = 1.0
mesh_optimization_iters = 2
mesh_filename = "c5g7_UO2_pin.inp"

# Geometry
gmsh.initialize()
gmsh.model.occ.add_disk(pin_pitch/2, pin_pitch/2, 0, pin_radius, pin_radius)
gmsh.model.occ.synchronize()

# Assign materials
# TODO: Update to use name= optional arg when gmsh 4.10 comes out
ptag = gmsh.model.add_physical_group(2, [1])
gmsh.model.set_physical_name(2, ptag, "MATERIAL_UO2")

# Overlay grid
core_bb = lattice_bb = raytracing_module = pin_bb = AAB(pin_pitch, pin_pitch)
cell = MPACTCoarseCell(pin_bb) 
lattice = MPACTLattice
coarse_cell_grid = [[pin_bb;;];;]
MPACT_grid = MPACTGridOverlay2D(core_bb, 
                                lattice_grid, 
                                raytracing_module, 
                                coarse_cell_grid) 



gmsh_overlay_rectangular_grid(bounding_box, grid_material, grid_nx, grid_ny)

# Visualize geometry prior to meshing
# gmsh.fltk.run()

# Mesh
gmsh.model.mesh.set_size(gmsh.model.get_entities(0), mesh_char_len)
gmsh.model.mesh.generate(2)

# Optimize the mesh
for n in 1:mesh_optimization_iters
    gmsh.model.mesh.optimize("Laplace2D")
end

# Visualize the mesh
gmsh.fltk.run()

# Write the mesh to file 
gmsh.write(mesh_filename)

# Finalize gmsh
gmsh.finalize()

# Read the abaqus mesh into a 2D unstructured mesh data structure
mesh = read_abaqus_2d(mesh_filename)
