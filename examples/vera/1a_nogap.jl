# VERA Core Physics Benchmark Progression Problem Specifications
# Revision 4, August 29, 2014
# CASL-U-2012-0131-004
# k-eff = 1.187038 ± 0.000054 (pg. 23)
using UM2
file_prefix = "1a"
mesh_order = 2
mesh_faces = "Triangle"
lc = 1.26/3 # cm, pitch = 1.26 cm
lc_str = string(round(lc, digits = 4))
full_file_prefix = file_prefix * "_" * lowercase(mesh_faces) * string(mesh_order) * 
                   "_" * replace(lc_str, "."=>"_") 
add_timestamps_to_log()
gmsh.initialize()
gmsh.option.set_number("General.NumThreads", 0) # 0 uses system default, i.e. OMP_NUM_THREADS)
gmsh.option.set_number("Geometry.OCCParallel", 1) # use parallel OCC boolean operations
gmsh.option.set_number("General.Verbosity", 2) # 1: +errors, 2: +warnings, 3: +direct, 4: +information, 5: +status, 99: +debug

# Model
# ----------------------------------------------------------------------------------------------
@info "Creating model"
fuel_entities = Int64[]
clad_entities = Int64[]
water_entities = Int64[]
r_fuel = 0.4096 # Pellet radius = 0.4096 cm (pg. 4)
r_clad = 0.475  # Outer clad radius = 0.475 cm (pg.4)
pitch = 1.26    # Pitch = 1.26 cm (pg. 4)

x = pitch / 2
y = pitch / 2
push!(fuel_entities, gmsh.model.occ.add_disk(x, y, 0, r_fuel, r_fuel))
push!(clad_entities, gmsh.model.occ.add_disk(x, y, 0, r_clad, r_clad))
gmsh.model.occ.synchronize()

# Materials
# ----------------------------------------------------------------------------------------------
# http://juliagraphics.github.io/Colors.jl/stable/namedcolors/
materials = Material[Material(name = "Fuel", color = "forestgreen"),
                     Material(name = "Clad", color = "slategrey")]
safe_add_physical_group("Material: Fuel", [(2, i) for i in fuel_entities])
safe_add_physical_group("Material: Clad", [(2, i) for i in clad_entities])
ents = gmsh.model.get_entities(2)
# Fragment the many disks, prioritizing Fuel > Gap > Clad to fill space
safe_fragment(ents, ents, material_hierarchy = materials)

# Overlay Grid
# ---------------------------------------------------------------------------------------------
coarse_divs = [0.0, pitch]
coarse_grid = RectilinearGrid(coarse_divs, coarse_divs)
mpact_grid = MPACTSpatialPartition(coarse_grid)
# We now want water to fill empty space, preserving all other materials,
# so we need to add water to the bottom of the materials hierarchy
push!(materials, Material(name = "Water", color = "royalblue"))
overlay_mpact_grid_hierarchy(mpact_grid, materials)

# Mesh
# ------------------------------------------------------------------------------------------------
mat_lc = [(mat, lc) for mat in materials]    
set_mesh_field_by_material(mat_lc)   
generate_mesh(order = mesh_order, face_type = mesh_faces, opt_iters = 2, force_quads = true)
gmsh.write(full_file_prefix*".inp")
mesh_errors = get_cad_to_mesh_errors()    
for i in eachindex(mesh_errors)    
    println(mesh_errors[i])    
end    
gmsh.fltk.run()
gmsh.finalize()

mat_names, mats, elsets, mesh = import_mesh(full_file_prefix*".inp")    
# Partition the mesh according to mpact's spatial hierarchy.    
leaf_elsets, hierarchical_mesh = partition_mesh(mesh, elsets, by = "MPACT")
# Write the mesh to an xdmf file
export_mesh(hierarchical_mesh, leaf_elsets, full_file_prefix*".xdmf")
