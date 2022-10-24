using UM2
mesh_order = 1
mesh_face = "Triangle"


cell_pitch = 8.0
lc = cell_pitch / 8 

add_timestamps_to_log()
gmsh.initialize()
# 1: +errors, 2: +warnings, 3: +direct, 4: +information, 
# 5: +status, 99: +debug
gmsh.option.set_number("General.Verbosity", 4)

# Model
# ----------------------------------------------------------------------------------------------
# 2 x 2 assemblies
# 2 x 2 modules per assembly
# 2 x 2 coarse cells per module
# 8 by 8 coarse cells in the core 
@info "Creating model"
mod_pitch = 2 * cell_pitch
ass_pitch = 2 * mod_pitch
disk_radius = cell_pitch / (8 * 2)
disk_entities = Int64[];

ctr = 0
for ja in 1:2, ia in 1:2
    for jmo in 1:2, imo in 1:2
        for jc in 1:2, ic in 1:2
            x = (ia  - 1) * ass_pitch + 
                (imo - 1) * mod_pitch + 
                (ic  - 1) * cell_pitch
            y = (ja  - 1) * ass_pitch + 
                (jmo - 1) * mod_pitch + 
                (jc  - 1) * cell_pitch
            for j in UInt32(1):UInt32(8), i in UInt32(1):UInt32(8)
                if morton_encode(i - UInt32(1), j - UInt32(1)) ≤ ctr
                push!(disk_entities, gmsh.model.occ.add_disk(x + (2i - 1) * disk_radius, 
                                                             y + (2j - 1) * disk_radius, 
                                                             0.0, 
                                                             disk_radius, disk_radius))
                end
            end
            global ctr += 1
        end
    end
end
gmsh.model.occ.synchronize()
#gmsh.fltk.run()

# Materials
# ----------------------------------------------------------------------------------------------
# http://juliagraphics.github.io/Colors.jl/stable/namedcolors/
materials = Material[Material(name = "Fuel", color = "forestgreen"),
                     Material(name = "Water", color = "royalblue")]
safe_add_physical_group("Material: Fuel", [(2, i) for i in disk_entities])
ents = gmsh.model.get_entities(2)

# Overlay Grid
# ---------------------------------------------------------------------------------------------
core = RectilinearPartition2(UM_I(1), 
                             "Core", 
                             RectilinearGrid(UM_F[0, ass_pitch, 2*ass_pitch],
                                             UM_F[0, ass_pitch, 2*ass_pitch]),
                             UM_I[1 3; 2 4])
lattices = Vector{RegularPartition2{UM_I}}(undef, 4)
ass_ctr = 0
for ja in 1:2, ia in 1:2
    global ass_ctr += 1
    ass_name = "Lattice_0000" * string(ass_ctr)
    minima = Point((ia - 1) * ass_pitch, (ja - 1) * ass_pitch)
    ndiv = (UM_I(2), UM_I(2))
    lat_part = RegularPartition2(UM_I(ass_ctr),
                                ass_name,
                                RegularGrid(minima, (mod_pitch, mod_pitch), ndiv),
                                UM_I[4 * (ass_ctr - 1) + 1 4 * (ass_ctr - 1) + 3;
                                     4 * (ass_ctr - 1) + 2 4 * (ass_ctr - 1) + 4])
    lattices[ass_ctr] = lat_part
end
    
modules = Vector{RectilinearPartition2{UM_I}}(undef, 16)
mod_ctr = 0
for ja in 1:2, ia in 1:2
    for jmo in 1:2, imo in 1:2
        global mod_ctr += 1
        if mod_ctr < 10
            mod_name = "Module_0000" * string(mod_ctr)
        else
            mod_name = "Module_000" * string(mod_ctr)
        end
        x = (ia  - 1) * ass_pitch + (imo - 1) * mod_pitch
        y = (ja  - 1) * ass_pitch + (jmo - 1) * mod_pitch 
        grid = RectilinearGrid(UM_F[x, x + cell_pitch, x + 2 * cell_pitch], 
                               UM_F[y, y + cell_pitch, y + 2 * cell_pitch])
        mat = UM_I[4 * (mod_ctr - 1) + 1 4 * (mod_ctr - 1) + 3;
                   4 * (mod_ctr - 1) + 2 4 * (mod_ctr - 1) + 4]
        mod_part = RectilinearPartition2(UM_I(mod_ctr), 
                                            mod_name, 
                                            grid, 
                                            mat)
        modules[mod_ctr] = mod_part
    end
end
mpact_grid = MPACTSpatialPartition(core, lattices, modules)
overlay_mpact_grid_hierarchy(mpact_grid, materials)
#gmsh.fltk.run()

# Mesh
# ------------------------------------------------------------------------------------------------
mat_lc = [(mat, lc) for mat in materials]
set_mesh_field_by_material(mat_lc)
generate_mesh(order = mesh_order, face_type = mesh_face, opt_iters = 2, force_quads = false)
gmsh.write("map.inp")
mesh_errors = get_cad_to_mesh_errors()
for i in eachindex(mesh_errors)
    println(mesh_errors[i])
end
#gmsh.fltk.run()
gmsh.finalize()

mat_names, mats, elsets, mesh = import_mesh("map.inp")
leaf_elsets, hierarchical_mesh = partition_mesh(mesh, elsets, by = "MPACT")
export_mesh(hierarchical_mesh, leaf_elsets, "map.xdmf")
