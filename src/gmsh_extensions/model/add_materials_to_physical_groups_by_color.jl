"""
    add_materials_to_physical_groups_by_color(
        mat_to_color::Dict{String, NTuple{4, Int64}}, 
        color_to_ent::Dict{NTuple{4, Int32}, Vector{Int32}},
        dim::Int64=2)

Assign materials to entities of dimension `dim`, mapping material name -> 
color -> entities. Note, each material name must contain "Material", but is
not case-sensitive.

See `gmsh.model.get_entities_by_color`.
"""
function add_materials_to_physical_groups_by_color(
            materials::Vector{Material},
            color_to_ent::Dict{NTuple{4, Int32}, Vector{Int32}},
            dim::Int64=2)

    for mat in sort!(materials, by = m -> m.name)
        tags = color_to_ent[mat.color]
        p = gmsh.model.add_physical_group(dim, tags)
        gmsh.model.set_physical_name(dim, p, "Material: "*mat.name)
    end
    return nothing 
end

