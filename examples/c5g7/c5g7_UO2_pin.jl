using MOCNeutronTransport

# Parameters
bounding_box = (0.0, 1.26, 0.0, 1.26) # (xmin, xmax, ymin, ymax)
pin_pitch = 1.26 # cm
pin_radius = 0.54 # cm
grid_material = "MATERIAL_WATER"
grid_nx = [1]
grid_ny = [1]
mesh_char_len = 0.2
mesh_optimization_iters = 2
mesh_filename = "c5g7_UO2_pin.inp"

# Initialize msh
gmsh.initialize()

# Geometry
gmsh.model.occ.add_disk(pin_pitch/2.0, pin_pitch/2.0, 0.0, pin_radius, pin_radius)
gmsh.model.occ.synchronize()

# Assign materials
ptag = gmsh.model.add_physical_group(2, [1])
gmsh.model.set_physical_name(2, ptag, "MATERIAL_UO2-3.3")

# Overlay grid
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
