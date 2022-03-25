"""
    add_physical_group(name::String, entities::Vector{Tuple{Int32, Int32}})

A combined `gmsh.model.add_physical_group` and `gmsh.model.set_physical_name` that
accounts for physical groups that already exist with name `name`.
"""
function add_physical_group(name::String, entities::Vector{Tuple{Int32, Int32}})
    dims = unique(getindex.(entities, 1))
    for dim in dims
        current_groups = gmsh.model.get_physical_groups(dim)
        current_names = [gmsh.model.get_physical_name(dim, grp[2]) for grp in current_groups]
        if name ∈ current_names
            id = findfirst(x->x == name, current_names)
            tag = current_groups[id][2]
            tags = gmsh.model.get_entities_for_physical_group(dim, tag)
            append!(entities, [(dim, t) for t in tags])
            unique!(entities)
            gmsh.model.remove_groups([(dim, tag)])
        end
        p = gmsh.model.add_physical_group(dim, 
                                         [ent[2] for ent in filter(e->e[1] == dim, entities)]
                                         )
        gmsh.model.set_physical_name(dim, p, name)
    end
    return nothing
end
