"""
    add_materials_to_physical_groups_by_color(
        materials::Vector{Material},
        color_to_ent::Dict{NTuple{4, Int32}, Vector{NTuple{2, Int32}}}
        )

Assign materials to entities, mapping material name -> color -> entities. 
"""
function add_materials_to_physical_groups_by_color(
        materials::Vector{Material},
        color_to_ent::Dict{NTuple{4, Int32}, Vector{NTuple{2, Int32}}}
    )
    for mat in sort!(materials, by = m -> m.name)
        ents = color_to_ent[mat.color]
        for i = 0:3
            e_of_dim_i = filter(e->e[1] == i, ents)
            p = gmsh.model.add_physical_group(i, getindex.(e_of_dim_i, 2))
            gmsh.model.set_physical_name(i, p, "Material: "*mat.name)
        end
    end
    return nothing 
end
