"""
    gmsh.model.add_cad_entity_names_to_physical_groups(dim::Int64=-1)

Add the names of imported CAD entities of dimension `dim` to the model's
physical groups. For an entity with name "Shapes/X/Y/Z", the physical group it
will be assigned is "Z".
"""
function add_cad_entity_names_to_physical_groups(dim::Int64=-1)
    physicals = Dict{String, Vector{Tuple{Int32, Int32}}}()
    for ent in gmsh.model.get_entities(dim)
        edim, tag = ent
        name = gmsh.model.get_entity_name(edim, tag)
        if name != ""
            path = split(name, "/")
            ent_name = path[end]
            if ent_name ∉ keys(physicals)
                physicals[ent_name] = Tuple{Int32, Int32}[]
            end
            push!(physicals[ent_name], ent)
        end
    end
    current_physical_groups = gmsh.model.get_physical_groups(dim)
    current_physical_names = String[]
    for dim_tag in current_physical_groups
        push!(current_physical_names, gmsh.model.get_physical_name(dim_tag[1], dim_tag[2]))
    end
    for (name, ents) in sort(collect(physicals))
        for i = 0:3
            e_of_dim_i = filter(e->e[1] == i, ents)
            p = gmsh.model.add_physical_group(i, getindex.(e_of_dim_i, 2))
            gmsh.model.set_physical_name(i, p, name)
        end
    end
    return sort!(collect(keys(physicals)))
end
