export safe_add_physical_group

"""
    safe_add_physical_group(name::String, entities::Vector{NTuple{2,Int32}})

A combined `gmsh.model.add_physical_group` and `gmsh.model.set_physical_name` that
accounts for physical groups that already exist with name `name`. 

Returns the tag of the physical group.
"""
function safe_add_physical_group(name::String, entities::Vector{NTuple{2,I}}) where {I<:Integer}
    dims = unique!(getindex.(entities, 1))
    for dim in dims
        current_groups = gmsh.model.get_physical_groups(dim)
        current_names = [gmsh.model.get_physical_name(dim, grp[2]) for grp in current_groups]
        if name ∈ current_names
            id = findfirst(x->x == name, current_names)
            tag = current_groups[id][2]
            tags = gmsh.model.get_entities_for_physical_group(dim, tag)
            append!(entities, [(dim, t) for t in tags])
            unique!(entities)
            gmsh.model.remove_physical_groups([(dim, tag)])
            p = gmsh.model.add_physical_group(dim, 
                                             [ent[2] for ent in filter(e->e[1] == dim, entities)],
                                             tag
                                             )
        else
            p = gmsh.model.add_physical_group(dim, 
                                             [ent[2] for ent in filter(e->e[1] == dim, entities)]
                                             )
        end
        gmsh.model.set_physical_name(dim, p, name)
    end
    return p
end
