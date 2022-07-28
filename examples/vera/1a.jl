# VERA Core Physics Benchmark Progression Problem Specifications
# Revision 4, August 29, 2014
# CASL-U-2012-0131-004
# k-eff = 1.187038 ± 0.000054 (pg. 23)
using UM2
file_prefix = "1a"
mesh_order = 2
mesh_faces = "Quadrilateral"
lc = 1.26/5 # cm, pitch = 1.26 cm
lc_str = string(round(lc, digits = 4))
full_file_prefix = file_prefix * "_" * lowercase(mesh_faces) * string(mesh_order) * 
                   "_" * replace(lc_str, "."=>"_") 

add_timestamps_to_logger()
gmsh.initialize()
gmsh.option.set_number("General.NumThreads", 0) # 0 uses system default, i.e. OMP_NUM_THREADS)
gmsh.option.set_number("Geometry.OCCParallel", 1) # use parallel OCC boolean operations
gmsh.option.set_number("General.Verbosity", 2) # 1: +errors, 2: +warnings, 3: +direct, 4: +information, 5: +status, 99: +debug

# Model
# ----------------------------------------------------------------------------------------------
fuel_entities = Int64[];
gap_entities = Int64[];
clad_entities = Int64[];
water_entities = Int64[];
r_fuel = 0.4096 # Pellet radius = 0.4096 cm (pg. 4)
r_gap = 0.418   # Inner clad radius = 0.418 cm (pg.4)
r_clad = 0.475  # Outer clad radius = 0.475 cm (pg.4)
pitch = 1.26    # Pitch = 1.26 cm (pg. 4)

x = pitch / 2
y = pitch / 2
push!(fuel_entities, gmsh.model.occ.add_disk(x, y, 0, r_fuel, r_fuel))
push!(gap_entities, gmsh.model.occ.add_disk(x, y, 0, r_gap, r_gap))
push!(clad_entities, gmsh.model.occ.add_disk(x, y, 0, r_clad, r_clad))
gmsh.model.occ.synchronize()

# Materials
# ----------------------------------------------------------------------------------------------
# http://juliagraphics.github.io/Colors.jl/stable/namedcolors/
materials = Material[Material(name = "Fuel", color = "forestgreen"),
                     Material(name = "Gap", color = "darkorange"),
                     Material(name = "Clad", color = "slategrey")]
safe_add_physical_group("Material: Fuel", [(2, i) for i in fuel_entities])
safe_add_physical_group("Material: Gap", [(2, i) for i in gap_entities])
safe_add_physical_group("Material: Clad", [(2, i) for i in clad_entities])
ents = gmsh.model.get_entities(2)
# Fragment the many disks, prioritizing Fuel > Gap > Clad to fill space
safe_fragment(ents, ents, material_hierarchy = materials)

# Overlay Grid
# ---------------------------------------------------------------------------------------------
coarse_divs = Vec(0.0, pitch)
coarse_grid = RectilinearGrid(coarse_divs, coarse_divs)
mpact_grid = MPACTGridHierarchy(coarse_grid)
# We now want water to fill empty space, preserving all other materials,
# so we need to add water to the bottom of the materials hierarchy
push!(materials, Material(name = "Water", color = "royalblue"))
overlay_mpact_grid_hierarchy(mpact_grid, materials)

# Mesh
# ------------------------------------------------------------------------------------------------
for mat in materials
    mat.lc = lc
end
set_mesh_field_using_materials(materials)
generate_mesh(order = mesh_order, faces = mesh_faces, opt_iters = 2, force_quads = false)
gmsh.write(full_file_prefix*".inp")
mesh_error = get_cad_to_mesh_error()
for i in eachindex(mesh_error)
    println(mesh_error[i])
end
gmsh.fltk.run()
gmsh.finalize()

mesh = import_mesh(full_file_prefix*".inp")
# Partition mesh according to mpact grid hierarchy and write as an xdmf file 
mpt = MeshPartitionTree(mesh)
export_mesh(full_file_prefix*".xdmf", mpt)
