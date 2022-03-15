# Lewis, E. E., et al. "Benchmark specification for Deterministic 2-D/3-D MOX fuel 
# assembly transport calculations without spatial homogenization (C5G7 MOX)." 
# NEA/NSC 280 (2001): 2001.
using MOCNeutronTransport

# Geometry
# ---------------------------------------------------------------------------------------
gmsh.initialize()
gmsh.merge("c5g7.step")

# Materials
# ---------------------------------------------------------------------------------------
# TODO: Update to use name= optional arg when gmsh 4.10 comes out
ptag = gmsh.model.add_physical_group(2, [1])
gmsh.model.set_physical_name(2, ptag, "MATERIAL_UO2")

# Overlay grid
# ---------------------------------------------------------------------------------------
# Finemesh
core_bb = lattice_bb = raytracing_module = pin_bb = AAB(pin_pitch, pin_pitch)
# A single pin cell
cell = MPACTCoarseCell(pin_bb) 
# A single ray tracing module
rt_module = MPACTRayTracingModule(raytracing_module, [cell.id;;])
# A single 1x1 lattice
lattice = MPACTLattice(lattice_bb, [rt_module.id;;])
core = MPACTCore2D(core_bb, [lattice.id;;])

gmsh_overlay_rectangular_grid(bounding_box, grid_material, grid_nx, grid_ny)

# Visualize geometry prior to meshing
# gmsh.fltk.run()

# Use incomplete elements.
# Use quad remesh experimental option?
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
