using MOCNeutronTransport
# VERA Core Physics Benchmark Progression Problem Specifications
# Revision 4, August 29, 2014
# CASL-U-2012-0131-004
#
# Model
# ----------------------------------------------------------------------------------------------
gmsh.initialize()
uo2_entities = Int32[]
gap_entities = Int32[]
clad_entities = Int32[]
h2o_entities = Int32[]
# Assembly pitch = 21.50 cm (pg. 7)
bb = (0.0, 21.5, 0.0, 21.5)

# UO2 pins
# ----------------------------------------------------------------------------------------------
# Pellet radius = 0.4096 cm (pg. 4)
# Inner clad radius = 0.418 cm (pg.4)
# Outer clad radius = 0.475 cm (pg.4)
# Pitch = 1.26 cm (pg. 4)
r_fuel = 0.4096
r_gap = 0.418
r_clad = 0.475
pitch = 1.26
# Place pins
# Inter-Assembly Half Gap  = 0.04 cm (pg. 7 or deduction from assembly/pin pitch)
# Guide tube locations (i, j) (pg. 5)
coords_gt = [ 
    (6, 15),
    (9, 15),
    (12, 15),
    (4, 14),
    (14, 14),
    (3, 12),
    (6, 12),
    (9, 12),
    (12, 12),
    (15, 12),
    (3, 9), 
    (6, 9), 
    (12, 9), 
    (15, 9), 
    (3, 6), 
    (6, 6), 
    (9, 6), 
    (12, 6), 
    (15, 6), 
    (4, 4), 
    (14, 4), 
    (6, 3), 
    (9, 3), 
    (12, 3), 
]
# Instrument tube locations (pg. 5)
coords_it = [(9,9)]
for i = 1:17
    for j = 1:17
        if (i, j) ∈  coords_gt || (i,j) ∈  coords_it
          continue
        end
        push!(uo2_entities, 
              gmsh.model.occ.add_disk(0.04 + i*pitch - pitch/2, 
                                      0.04 + j*pitch - pitch/2, 
                                      0, r_fuel, r_fuel))
        push!(gap_entities, 
              gmsh.model.occ.add_disk(0.04 + i*pitch - pitch/2, 
                                      0.04 + j*pitch - pitch/2, 
                                      0, r_gap, r_gap))
        push!(clad_entities, 
              gmsh.model.occ.add_disk(0.04 + i*pitch - pitch/2, 
                                      0.04 + j*pitch - pitch/2, 
                                      0, r_clad, r_clad))
    end
end
# Guide tubes
# ----------------------------------------------------------------------------------------------
# Inner Guide Tube Radius = 0.561 cm (pg. 5)
# Outer Guide Tube Radius = 0.602 cm (pg. 5)
r_gt_inner = 0.561
r_gt_outer = 0.602
for (i, j) in coords_gt
    push!(h2o_entities,
          gmsh.model.occ.add_disk(0.04 + i*pitch - pitch/2, 
                                  0.04 + j*pitch - pitch/2, 
                                  0, r_gt_inner, r_gt_inner))
    push!(clad_entities, 
          gmsh.model.occ.add_disk(0.04 + i*pitch - pitch/2, 
                                  0.04 + j*pitch - pitch/2, 
                                  0, r_gt_outer, r_gt_outer))
end
# Instrument tube
# ----------------------------------------------------------------------------------------------
# Inner Instrument Tube Radius = 0.559 cm (pg. 5)
# Outer Instrument Tube Radius = 0.605 cm (pg. 5)
r_it_inner = 0.559
r_it_outer = 0.605
for (i, j) in coords_it
    push!(h2o_entities,
          gmsh.model.occ.add_disk(0.04 + i*pitch - pitch/2, 
                                  0.04 + j*pitch - pitch/2, 
                                  0, r_it_inner, r_it_inner))
    push!(clad_entities, 
          gmsh.model.occ.add_disk(0.04 + i*pitch - pitch/2, 
                                  0.04 + j*pitch - pitch/2, 
                                  0, r_it_outer, r_it_outer))
end
gmsh.model.occ.synchronize()
# Materials
# ----------------------------------------------------------------------------------------------
gmsh.model.occ.synchronize()
p = gmsh.model.add_physical_group(2, uo2_entities)
gmsh.model.set_physical_name(2, p, "MATERIAL_UO2")
p = gmsh.model.add_physical_group(2, gap_entities)
gmsh.model.set_physical_name(2, p, "MATERIAL_GAP")
p = gmsh.model.add_physical_group(2, h2o_entities)
gmsh.model.set_physical_name(2, p, "MATERIAL_WATER")
p = gmsh.model.add_physical_group(2, clad_entities)
gmsh.model.set_physical_name(2, p, "MATERIAL_CLAD")
gmsh_group_preserving_fragment(gmsh.model.get_entities(2),
                               gmsh.model.get_entities(2);
                               material_hierarchy = ["MATERIAL_UO2", 
                                                     "MATERIAL_GAP",
                                                     "MATERIAL_WATER",
                                                     "MATERIAL_CLAD"])

# Overlay Grid
# ---------------------------------------------------------------------------------------------
x = [[pitch*i + 0.04 for i = 1:16]]
grid_tags = gmsh_overlay_rectangular_grid(bb, "MATERIAL_WATER", x, x) 
# gmsh.fltk.run()

# Mesh
# ------------------------------------------------------------------------------------------------
lc = 0.3 # cm
gmsh.model.mesh.set_size(gmsh.model.get_entities(0), lc)
# Optional mesh optimization:
niter = 2 # The optimization iterations

# Triangles
# gmsh.model.mesh.generate(2) # 2 is dimension of mesh
# for () in 1:niter
#     gmsh.model.mesh.optimize("Laplace2D")
# end

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
# gmsh.option.set_number("Mesh.HighOrderOptimize", 2)
# gmsh.model.mesh.generate(2) # Triangles first for high order meshes.
# gmsh.model.mesh.set_order(2)
## for () in 1:niter
##     gmsh.model.mesh.optimize("HighOrderElastic")
##     gmsh.model.mesh.optimize("Relocate2D")
##     gmsh.model.mesh.optimize("HighOrderElastic")
## end

# 2nd order quadrilaterals
# These can be problematic for large lc. They have trouble respecting CAD boundaries.

gmsh.option.set_number("Mesh.RecombineAll", 1) # recombine all triangles
gmsh.option.set_number("Mesh.Algorithm", 8) # Frontal-Delaunay for quads. Better 2D algorithm
gmsh.option.set_number("Mesh.RecombinationAlgorithm", 1)
gmsh.option.set_number("Mesh.HighOrderOptimize", 2)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.set_order(2)
## for () in 1:niter
##     gmsh.model.mesh.optimize("HighOrderElastic")
##     gmsh.model.mesh.optimize("Relocate2D")
##     gmsh.model.mesh.optimize("HighOrderElastic")
## end

# gmsh.fltk.run()

# Mesh conversion
#---------------------------------------------------------------------------------------------------
# We want to write an Abaqus file from Gmsh, read the Abaqus file into Julia, and convert that into
# a rectangularly partitioned hierarchical XDMF file, the final input to MPACT
#
# Write Abaqus file
gmsh.write("2a.inp")
gmsh.finalize() # done with Gmsh. Finalize
mesh = read_abaqus_2d("2a.inp")
# Convert mesh into Hierarchical Rectangularly Partitioned Mesh 
HRPM = partition_rectangularly(mesh)
write_xdmf_2d("2a.xdmf", HRPM)

# Mass conservation
# --------------------------------------------------------------------------------------------------
total_area_ref = 21.5^2
uo2_area_ref = π*(r_fuel^2)*(17^2 - 25)
gap_area_ref = π*(r_gap^2 - r_fuel^2)*(17^2 - 25)
clad_area_ref = π*( (r_clad^2 - r_gap^2)*(17^2 - 25) +
                    (r_gt_outer^2 - r_gt_inner^2)*(24) +
                    (r_it_outer^2 - r_it_inner^2)*(1) )
h2o_area_ref = total_area_ref - uo2_area_ref - gap_area_ref - clad_area_ref

uo2_area = area(mesh, "MATERIAL_UO2")
println("UO₂ area (reference): $uo2_area_ref")
println("UO₂ area  (computed): $uo2_area")
err = 100*(uo2_area - uo2_area_ref)/uo2_area_ref
println("Error     (relative): $err %") 
println("")

gap_area = area(mesh, "MATERIAL_GAP")
println("Gap area (reference): $gap_area_ref")
println("Gap area  (computed): $gap_area")
err = 100*(gap_area - gap_area_ref)/gap_area_ref
println("Error     (relative): $err %") 
println("")

clad_area = area(mesh, "MATERIAL_CLAD")
println("Clad area (reference): $clad_area_ref")
println("Clad area  (computed): $clad_area")
err = 100*(clad_area - clad_area_ref)/clad_area_ref
println("Error      (relative): $err %") 
println("")

h2o_area = area(mesh, "MATERIAL_WATER")
println("H₂O area (reference): $h2o_area_ref")
println("H₂O area  (computed): $h2o_area")
err = 100*(h2o_area - h2o_area_ref)/h2o_area_ref
println("Error     (relative): $err %") 
println("")

total_area = uo2_area + gap_area + clad_area + h2o_area 
println("Total area (reference): $total_area_ref")
println("Total area  (computed): $total_area")
err = 100*(total_area - total_area_ref)/total_area_ref
println("Error     (relative): $err %") 
println("")
