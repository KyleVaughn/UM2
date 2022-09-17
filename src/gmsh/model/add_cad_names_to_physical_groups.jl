export add_cad_names_to_physical_groups

"""
    add_cad_names_to_physical_groups(dim::Int64 = -1)

Add the names of imported CAD entities of dimension `dim` to the model's
physical groups. For an entity with name "Shapes/X/Y/Z", the physical group it
will be assigned is "Z". If `dim` == -1, add the names of all entities.
"""
function add_cad_names_to_physical_groups(dim::Int64 = -1)
    # Get all entities. If they have a name, add them to the appropriate 
    # entry in the physicals dict
    physicals = Dict{String, Vector{NTuple{2, Int32}}}()
    for ent in gmsh.model.get_entities(dim)
        edim, tag = ent
        name = gmsh.model.get_entity_name(edim, tag)
        if name != ""
            path = split(name, "/")
            ent_name = path[end]
            if ent_name ∉ keys(physicals)
                physicals[ent_name] = NTuple{2, Int32}[]
            end
            push!(physicals[ent_name], ent)
        end
    end
    # For each of the groups, add them to the model
    for (name, ents) in pairs(physicals)
        safe_add_physical_group(name, ents)
    end
    return sort!(collect(keys(physicals)))
end
