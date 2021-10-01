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
    gmsh.model.occ.add_disk(0.04 + i*pitch - pitch/2, 
                            0.04 + j*pitch - pitch/2, 
                            0, r_gt_inner, r_gt_inner)
    push!(clad_entities, 
          gmsh.model.occ.add_disk(0.04 + i*pitch - pitch/2, 
                                  0.04 + j*pitch - pitch/2, 
                                  0, r_gt_outer, r_gt_outer))
end












gmsh.model.occ.synchronize()
gmsh.fltk.run()
# Materials
# ----------------------------------------------------------------------------------------------
gmsh.model.occ.synchronize()
p = gmsh.model.add_physical_group(2, uo2_entities)
gmsh.model.set_physical_name(2, p, "MATERIAL_UO2")
p = gmsh.model.add_physical_group(2, gap_entities)
gmsh.model.set_physical_name(2, p, "MATERIAL_GAP")
p = gmsh.model.add_physical_group(2, clad_entities)
gmsh.model.set_physical_name(2, p, "MATERIAL_CLAD")
gmsh_group_preserving_fragment(gmsh.model.get_entities(2),
                               gmsh.model.get_entities(2);
                               material_hierarchy = ["MATERIAL_UO2", 
                                                     "MATERIAL_GAP",
                                                     "MATERIAL_CLAD"])
gmsh.fltk.run()
#
## Overlay Grid
## ---------------------------------------------------------------------------------------------
## The preferable locations to place a grid division are the horizontal or vertical lines that do
## not intersect any pins. This way, we can potentially capture one pin per coarse cell, and easily
## show pin powers.
##
## If we define the center of the model at (0, 0)
## The valid x or y locations for the UO2 pins in the positive x, y quadrant are:
## [0, pitch/2 - r_clad)
## ( [(2i-1)/2]pitch + r_clad, [(2(i+1)-1)/2]pitch - r_clad)
## (23pitch/2 + r_clad, ∞)
## i = 1, 2, ..., 11
## The valid x or y locations for the U metal pins are 
## [0, pitch_m/2 - r_clad_m)
## ( [(2i-1)/2]pitch_m + r_clad_m, [(2(i+1)-1)/2]pitch_m - r_clad_m)
## (21pitch_m/2 + r_clad_m, ∞)
## i = 1, 2, ..., 10
## Not that due to symmetry, -1*interval is also valid.
## We may compute the intersections of these sets.
#uo2_sets = [(0.0, pitch/2 - r_clad)]
#for i = 1:11
#    push!(uo2_sets, ( ((2i-1)/2)pitch + r_clad, ((2(i+1)-1)/2)pitch - r_clad))
#end
#push!(uo2_sets, (23pitch/2 + r_clad, Inf))
#
#um_sets = [(0.0, pitch_m/2 - r_clad_m)]
#for i = 1:10
#    push!(um_sets, ( ((2i-1)/2)pitch_m + r_clad_m, ((2(i+1)-1)/2)pitch_m - r_clad_m))
#end
#push!(um_sets, (21pitch_m/2 + r_clad_m, Inf))
#
#set_intersections = Tuple{Float64, Float64}[]
#for (uo2_min, uo2_max) in uo2_sets
#    for (um_min, um_max) in um_sets
#        # um set intersects on the left
#        if um_min ≤ uo2_min ≤ min(um_max, uo2_max)
#            push!(set_intersections, (uo2_min, min(um_max, uo2_max)))
#        # um set intersects on the right
#        elseif um_min ≤ uo2_max ≤ um_max
#            push!(set_intersections, (max(um_min, uo2_min), uo2_max))
#        end
#    end
#end
## Visualize the intervals in 1d
##for (ymin, ymax) in set_intersections
##    if ymax == Inf
##        ymax = bb_apothem
##    end
##    gmsh.model.occ.add_rectangle(bb_apothem, ymin + bb_apothem, 0.0, 30.0, ymax - ymin)
##end
##gmsh.model.occ.synchronize()
##
## Note that we may also use intervals of length 2*value, and divide the problem using an odd
## number of divisions
##
## Possible RT modules ± ϵ. Check the intervals for full range of valid values.
## 1 x 1 - (-65, 65)
## 2 x 2 - (-65, 0, 65)
## 3 x 3 - (-60, -20,  20, 60)
## 4 x 4 - (-65, -32.5, 0, 32.5, 65)
## 5 x 5 - (-56, -33.6, -11.2, 11.2, 33.6, 56) & (-73, -43.8, -14.6, 14.6, 43.8, 73) 
## 6 x 6 - None
## 7 x 7 - None
## 8 x 8 - None
## ... None
#nx = [5]
#ny = [5]
#grid_tags = gmsh_overlay_rectangular_grid(bb, "MATERIAL_MODERATOR", nx, ny) 
#gmsh.fltk.run()
#
gmsh.finalize()
