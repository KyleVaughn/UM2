using UM2
mesh_order = 1
mesh_face = "Triangle"
pitch = 1.0
lc = pitch/5 

add_timestamps_to_log()
gmsh.initialize()
# 1: +errors, 2: +warnings, 3: +direct, 4: +information, 
# 5: +status, 99: +debug
gmsh.option.set_number("General.Verbosity", 4)

# Model
# ----------------------------------------------------------------------------------------------
# 4 assemblies
# 4 modules per assembly
# 4 coarse cells per module
# 64 by 64 coarse cells in the core 
@info "Creating model"
disk_radius
disk_entities = Int64[];

# Guide tube locations (i, j) (pg. 5)
coords_gt = [(6, 15), (9, 15), (12, 15), (4, 14), (14, 14), (3, 12), (6, 12), (9, 12),
    (12, 12),
    (15, 12), (3, 9), (6, 9), (12, 9), (15, 9), (3, 6), (6, 6), (9, 6), (12, 6),
    (15, 6), (4, 4), (14, 4), (6, 3), (9, 3), (12, 3)]

# Instrument tube locations (pg. 5)
coords_it = [(9, 9)]

# Place UO₂ pins
for i in 1:17, j in 1:17
    if (i, j) ∈ coords_gt || (i, j) ∈ coords_it
        continue
    end
    x = half_gap + i * pitch - pitch / 2
    y = half_gap + j * pitch - pitch / 2
    push!(fuel_entities, gmsh.model.occ.add_disk(x, y, 0, r_fuel, r_fuel))
    push!(clad_entities, gmsh.model.occ.add_disk(x, y, 0, r_clad, r_clad))
end

# Place guide tubes
r_gt_inner = 0.561 # Inner Guide Tube Radius = 0.561 cm (pg. 5)
r_gt_outer = 0.602 # Outer Guide Tube Radius = 0.602 cm (pg. 5)
for (i, j) in coords_gt
    x = half_gap + i * pitch - pitch / 2
    y = half_gap + j * pitch - pitch / 2
    push!(water_entities, gmsh.model.occ.add_disk(x, y, 0, r_gt_inner, r_gt_inner))
    push!(clad_entities, gmsh.model.occ.add_disk(x, y, 0, r_gt_outer, r_gt_outer))
end

# Place instrument tube
r_it_inner = 0.559 # Inner Instrument Tube Radius = 0.559 cm (pg. 5)
r_it_outer = 0.605 # Outer Instrument Tube Radius = 0.605 cm (pg. 5)
for (i, j) in coords_it
    x = half_gap + i * pitch - pitch / 2
    y = half_gap + j * pitch - pitch / 2
    push!(water_entities, gmsh.model.occ.add_disk(x, y, 0, r_it_inner, r_it_inner))
    push!(clad_entities, gmsh.model.occ.add_disk(x, y, 0, r_it_outer, r_it_outer))
end
gmsh.model.occ.synchronize()

# Materials
# ----------------------------------------------------------------------------------------------
# http://juliagraphics.github.io/Colors.jl/stable/namedcolors/
materials = Material[Material(name = "Fuel", color = "forestgreen"),
                     Material(name = "Water", color = "royalblue"),
                     Material(name = "Clad", color = "slategrey")]
safe_add_physical_group("Material: Fuel", [(2, i) for i in fuel_entities])
safe_add_physical_group("Material: Water", [(2, i) for i in water_entities])
safe_add_physical_group("Material: Clad", [(2, i) for i in clad_entities])
ents = gmsh.model.get_entities(2)
# Fragment the many disks, prioritizing Fuel > Water > Clad to fill space
safe_fragment(ents, ents, material_hierarchy = materials)

# Overlay Grid
# ---------------------------------------------------------------------------------------------
coarse_divs = [pitch * i + 0.04 for i in 1:16]
pushfirst!(coarse_divs, 0.0)
push!(coarse_divs, 21.5) # Assembly pitch = 21.50 cm (pg. 7)
coarse_grid = RectilinearGrid(coarse_divs, coarse_divs)
mpact_grid = MPACTSpatialPartition(coarse_grid)
# We now want Water to fill empty space, preserving all other materials,
# so we need to swap Water to the bottom of the materials hierarchy
materials[2], materials[3] = materials[3], materials[2]
overlay_mpact_grid_hierarchy(mpact_grid, materials)

## Mesh
## ------------------------------------------------------------------------------------------------
mat_lc = [(mat, lc) for mat in materials]
set_mesh_field_by_material(mat_lc)
generate_mesh(order = mesh_order, face_type = mesh_face, opt_iters = 2, force_quads = false)
gmsh.write(full_file_prefix*".inp")
mesh_errors = get_cad_to_mesh_errors()
for i in eachindex(mesh_errors)
    println(mesh_errors[i])
end
#gmsh.fltk.run()
gmsh.finalize()
#
mat_names, mats, elsets, mesh = import_mesh(full_file_prefix*".inp")
###statistics(mesh)
# Partition the mesh according to mpact's spatial hierarchy.
leaf_elsets, hierarchical_mesh = partition_mesh(mesh, elsets, by = "MPACT")
# Write the mesh to an xdmf file
export_mesh(hierarchical_mesh, leaf_elsets, full_file_prefix*".xdmf")
