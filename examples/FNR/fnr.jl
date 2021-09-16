using MOCNeutronTransport

# I can't remember if the bottom left corner of the CMFD grid needs to be (0, 0, 0)
# To compensate, we do some extra math to ensure that happens.
grid_offset = 0.1 # Buffer between boundary of the model and the edge of the CMFD grid (cm)
# Bounding box for the CMFD grid
# xmin, xmax, ymin, ymax
bb = (0.0, 7.46252 + 2*grid_offset, 0.0, 8.31535 + 2*grid_offset) 


# Model setup
# ----------------------------------------------------------------------------------------
gmsh.initialize()
# Import CAD file
gmsh.merge("xsec.step")
# Visualize the model
# gmsh.fltk.run()

# Shift the model to the origin of the xy-plane
# Get the dim tags of all dimension 2 entites
dim_tags = gmsh.model.get_entities(2)
# Rotate the entities in dim_tags in the built-in CAD representation by angle radians 
# around the axis of revolution defined by the point (x, y, z) and the direction (ax, ay, az).
# gmsh.model.occ.rotate(dim_tags, x, y, z, ax, ay, az, angle)
gmsh.model.occ.rotate(dim_tags, 0, 1, 0, 0, 1, 0, -π/2)
# Translate it so that the corner is at the origin
gmsh.model.occ.translate(dim_tags, 37.3225 - 0.0099 + 10*grid_offset, 
                                   44.09 - 0.00174209 + 10*grid_offset, 
                                   -436.5625)
# Model is in mm. Convert to cm by shrinking everything by 1/10
gmsh.model.occ.dilate(dim_tags, 0, 0, 0, 1//10, 1//10, 0)
# Synchronize the CAD and Gmsh models
gmsh.model.occ.synchronize()
# Verify results
# gmsh.fltk.run()

# Assign materials
# ------------------------------------------------------------------------------------------
# Elements containing fuel were determined visually, since the numbering scheme isn't obvious
fuel_tags = [4, 5, 6, 9, 10, 11, 12, 13, 15, 20, 21, 25, 27, 31, 32, 33, 36, 38]
clad_tags = [t[2] for t in dim_tags]
# Determine which tags are clad by doing fuel\clad 
filter!(x->x ∉  fuel_tags, clad_tags)
# Assign the materials as physical groups
p = gmsh.model.add_physical_group(2, fuel_tags)
gmsh.model.set_physical_name(2, p, "MATERIAL_UO2")
p = gmsh.model.add_physical_group(2, clad_tags)
gmsh.model.set_physical_name(2, p, "MATERIAL_CLAD")
# Verify results
# gmsh.fltk.run()

# Overlay CMFD/hierarchical grid
# -------------------------------------------------------------------------------------------
# The arguments to gmsh_overlay_rectangular_grid are:
# (bb, material, nx, ny) or (bb, material, x, y)
#
# bb: The bounding box for the grid
#   bb = (xmin, xmax, ymin, ymax)::NTuple{4, AbstractFloat}
# material: The material with which to fill empty space/give to entities without a material
#   material::String
#   Note the material string must start with something that when converted to uppercase 
#       yields "MATERIAL_", ex: "Material_A" or "material_A"
# (nx, ny)/(x, y): The hierarchical grid divisions. 
#   The grid is mapped to MPACT in the following manner:
#       1st level: Assembly map
#       2nd level: Assemblies/lattices (same map, since 2D)
#       3rd level: Ray tracing modules (must be equal size)
#       4th level: Coarse cells in the CMFD grid
#   This has two input variations
#   Input 1) nx, ny
#       The number of equal x and y divisions in each level of the hierarchy.
#       nx::Vector{Int}, ny::Vector{Int}
#       Example:
#           nx = [3, 2, 1, 1]
#           The 1st level (assembly map) will be split into 3 pieces in x.
#           The 2nd level will split each of the 1st level grid elements into 2 piece in x
#               Theses elements are now 1/6 the total grid's x width
#           The 3rd level will be the same as above
#           The 4th level will be the same as above
#   Input 2) x, y
#       The x and y locations of the divisions. These need not be equally spaced.
#       x::Vector{Vector{T}}, y::Vector{Vector{T}}
#       Note that end points don't need to be included, and that any divisions in the coarser
#           grid levels are propagated downward.
#       Example:
#          For bb = (0, 1, 0, 1)
#          x = [[0.5],[0.2, 0.75]] 
#          is equal to
#          x = [[0.0, 0.5, 1.0], [0.0, 0.2, 0.5, 0.75, 1.0]] 
#   Final tidbit:
#       Any missing enties are assumed to be 1 division, where the last element in the input
#           vector is assumed to be the coarse cells.
#       Example:
#           nx = [1, 1, 4, 3] gives the same output as [4, 3]
#           4 RT modules (in x), 3 coarse cells (in x) per module = 12 total coarse cells (in x)
nx = [9, 1]
ny = [10, 1]
grid_tags = gmsh_overlay_rectangular_grid(bb, "MATERIAL_MODERATOR", nx, ny)
# Verify results
# gmsh.fltk.run()

# Mesh
# ------------------------------------------------------------------------------------------------
# Set the characteristic edge length of the mesh cells
lc = 0.3
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lc)
# Optional mesh optimization:
niter = 2 # The optimization iterations

# Triangles
gmsh.model.mesh.generate(2) # 2 is dimension of mesh
for () in 1:niter
    gmsh.model.mesh.optimize("Laplace2D")
end

# Quadrilaterals
# The default recombination algorithm might leave some triangles in the mesh, if
# recombining all the triangles leads to badly shaped quads. In such cases, to
# generate full-quad meshes, you can either subdivide the resulting hybrid mesh
# (with `Mesh.SubdivisionAlgorithm' set to 1), or use the full-quad
# recombination algorithm, which will automatically perform a coarser mesh
# followed by recombination, smoothing and subdivision.
# Mesh recombination algorithm (0: simple, 1: blossom, 2: simple full-quad, 3: blossom full-quad)
# Default = 1
# gmsh.option.set_number("Mesh.RecombineAll", 1) # recombine all triangles
# gmsh.option.set_number("Mesh.Algorithm", 8) # Frontal-Delaunay for quads. Better 2D algorithm
# gmsh.option.set_number("Mesh.RecombinationAlgorithm", 1)
# gmsh.model.mesh.generate(2)
# for () in 1:niter
#     gmsh.model.mesh.optimize("Laplace2D")
#     gmsh.model.mesh.optimize("Relocate2D")
#     gmsh.model.mesh.optimize("Laplace2D")
# end

# 2nd order triangles
# gmsh.model.mesh.generate(2) # Triangles first for high order meshes.
# gmsh.model.mesh.set_order(2)
# for () in 1:niter
#     gmsh.model.mesh.optimize("HighOrderElastic")
#     gmsh.model.mesh.optimize("Relocate2D")
#     gmsh.model.mesh.optimize("HighOrderElastic")
# end


# 2nd order quadrilaterals
# These can be problematic for large lc. They have trouble respecting CAD boundaries.
# gmsh.option.set_number("Mesh.RecombineAll", 1) # recombine all triangles
# gmsh.option.set_number("Mesh.Algorithm", 8) # Frontal-Delaunay for quads. Better 2D algorithm
# gmsh.option.set_number("Mesh.RecombinationAlgorithm", 3)
# gmsh.model.mesh.generate(2)
# gmsh.model.mesh.set_order(2)
# for () in 1:niter
#     gmsh.model.mesh.optimize("HighOrderElastic")
#     gmsh.model.mesh.optimize("Relocate2D")
#     gmsh.model.mesh.optimize("HighOrderElastic")
# end

gmsh.fltk.run()

# Mesh conversion
# ----------------------------------------------------------------------------------------------------
# We want to write an Abaqus file from Gmsh, read the Abaqus file into Julia, and convert that into
# a rectangularly partitioned hierarchical XDMF file, the final input to MPACT
#
# Write Abaqus file
gmsh.write("fnr.inp")
gmsh.finalize() # done with Gmsh. Finalize
mesh = read_abaqus_2d("fnr.inp")
# write_vtk_2d("fnr.vtk", mesh) # Can print vtk for sanity check. No cell sets
# write_xdmf_2d("fnr.xdmf", mesh) # Can print single grid XDMF
# Convert mesh into Hierarchical Rectangularly Partitioned Mesh 
HRPM = partition_rectangularly(mesh)
write_xdmf_2d("fnr.xdmf", HRPM)

# Mass conservation
# ----------------------------------------------------------------------------------------------------
# See how well your mesh conserves mass. Analytic solution not available, so fine tri6 mesh used as 
# reference
fuel_area = area(mesh, "MATERIAL_UO2")
fuel_area_ref = 6.119632103711788
println("Fuel area (reference): $fuel_area_ref")
println("Fuel area ( computed): $fuel_area")
err = 100*(fuel_area - fuel_area_ref)/fuel_area_ref
println("Error (relative): $err %") 
println("")

clad_area = area(mesh, "MATERIAL_CLAD")
clad_area_ref = 19.884367501087866
println("Clad area (reference): $clad_area_ref")
println("Clad area ( computed): $clad_area")
err = 100*(clad_area - clad_area_ref)/clad_area_ref
println("Error (relative): $err %") 
