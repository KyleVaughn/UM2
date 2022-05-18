# VERA Core Physics Benchmark Progression Problem Specifications
# Revision 4, August 29, 2014
# CASL-U-2012-0131-004
using MOCNeutronTransport

add_timestamps_to_logger()
gmsh.initialize()
gmsh.option.set_number("General.NumThreads", 0) # 0 uses system default, i.e. OMP_NUM_THREADS)
gmsh.option.set_number("Geometry.OCCParallel", 1) # use parallel OCC boolean operations
gmsh.option.set_number("General.Verbosity", 2) # 1: +errors, 2: +warnings, 3: +direct, 4: +information, 5: +status, 99: +debug

#
# Model
# ----------------------------------------------------------------------------------------------
uo2_entities = Int64[]; gap_entities = Int64[]; clad_entities = Int64[]; h2o_entities = Int64[]
r_fuel = 0.4096 # Pellet radius = 0.4096 cm (pg. 4)
r_gap = 0.418   # Inner clad radius = 0.418 cm (pg.4)
r_clad = 0.475  # Outer clad radius = 0.475 cm (pg.4)
pitch = 1.26    # Pitch = 1.26 cm (pg. 4)
half_gap = 0.04 # Inter-Assembly Half Gap  = 0.04 cm (pg. 7 or deduction from assembly/pin pitch)

# Guide tube locations (i, j) (pg. 5)
coords_gt = [ (6, 15), (9, 15), (12, 15), (4, 14), (14, 14), (3, 12), (6, 12), (9, 12), (12, 12),
              (15, 12), (3, 9), (6, 9), (12, 9),  (15, 9), (3, 6),  (6, 6),  (9, 6),  (12, 6),  
              (15, 6), (4, 4), (14, 4), (6, 3), (9, 3), (12, 3)]

# Instrument tube locations (pg. 5)
coords_it = [(9,9)]

# Place UO₂ pins
for i = 1:17, j = 1:17
    if (i, j) ∈  coords_gt || (i,j) ∈  coords_it
        continue
    end
    x = half_gap + i*pitch - pitch/2
    y = half_gap + j*pitch - pitch/2
    push!(uo2_entities, gmsh.model.occ.add_disk(x, y, 0, r_fuel, r_fuel))
    push!(gap_entities, gmsh.model.occ.add_disk(x, y, 0, r_gap, r_gap))
    push!(clad_entities, gmsh.model.occ.add_disk(x, y, 0, r_clad, r_clad))
end

# Place guide tubes
r_gt_inner = 0.561 # Inner Guide Tube Radius = 0.561 cm (pg. 5)
r_gt_outer = 0.602 # Outer Guide Tube Radius = 0.602 cm (pg. 5)
for (i, j) in coords_gt
    x = half_gap + i*pitch - pitch/2
    y = half_gap + j*pitch - pitch/2
    push!(h2o_entities, gmsh.model.occ.add_disk(x, y, 0, r_gt_inner, r_gt_inner))
    push!(clad_entities, gmsh.model.occ.add_disk(x, y, 0, r_gt_outer, r_gt_outer))
end

# Place instrument tube
r_it_inner = 0.559 # Inner Instrument Tube Radius = 0.559 cm (pg. 5)
r_it_outer = 0.605 # Outer Instrument Tube Radius = 0.605 cm (pg. 5)
for (i, j) in coords_it
    x = half_gap + i*pitch - pitch/2
    y = half_gap + j*pitch - pitch/2
    push!(h2o_entities, gmsh.model.occ.add_disk(x, y, 0, r_it_inner, r_it_inner))
    push!(clad_entities, gmsh.model.occ.add_disk(x, y, 0, r_it_outer, r_it_outer))
end
gmsh.model.occ.synchronize()

# Materials
# ----------------------------------------------------------------------------------------------
# http://juliagraphics.github.io/Colors.jl/stable/namedcolors/
materials = Material[
                Material(name="UO2", color="forestgreen"),
                Material(name="Gap", color="darkorange"),
                Material(name="H2O", color="royalblue"),
                Material(name="Clad", color="slategrey")
            ]
safe_add_physical_group("Material: UO2",  [(2,i) for i in uo2_entities])
safe_add_physical_group("Material: Gap",  [(2,i) for i in gap_entities])
safe_add_physical_group("Material: H2O",  [(2,i) for i in h2o_entities])
safe_add_physical_group("Material: Clad", [(2,i) for i in clad_entities])
ents = gmsh.model.get_entities(2)
# Fragment the many disks, prioritizing UO2 > Gap > H2O > Clad to fill space
safe_fragment(ents, ents, material_hierarchy = materials)
 
# Overlay Grid
# ---------------------------------------------------------------------------------------------
coarse_divs = [0.0]
append!(coarse_divs, [pitch*i + 0.04 for i = 1:16]) 
push!(coarse_divs, 21.5) # Assembly pitch = 21.50 cm (pg. 7)
scoarse_divs = Vec{18,Float64}(coarse_divs)
coarse_grid = RectilinearGrid(scoarse_divs, scoarse_divs)
mpact_grid = MPACTGridHierarchy(coarse_grid)
# We now want H2O to fill empty space, preserving all other materials,
# so we need to swap H2O to the bottom of the materials hierarchy
materials[3], materials[4] = materials[4], materials[3]
overlay_mpact_grid_hierarchy(mpact_grid, materials)

# Mesh
# ------------------------------------------------------------------------------------------------
lc = 0.25 # cm
for mat in materials
    mat.mesh_size = lc
end
set_mesh_field_using_materials(materials)
generate_mesh(order = 2, faces = "Triangle", opt_iters = 2)
gmsh.write("2a.inp")
mesh_error = get_cad_to_mesh_error()
for i in eachindex(mesh_error)
    println(mesh_error[i])
end
gmsh.fltk.run()
gmsh.finalize()

mesh = import_mesh("2a.inp")
statistics(mesh)
# Partition mesh according to mpact grid hierarchy, and write as an xdmf file 
mpt = partition(mesh)
export_mesh(mpt, "2a.xdmf")
