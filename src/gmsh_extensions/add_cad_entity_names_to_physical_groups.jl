@eval gmsh.model begin
    """
        add_cad_entity_names_to_physical_groups(dim::Int64=2)

    Add the names of imported CAD entities of dimension `dim` to the model's
    physical groups. For an entity with name "Shapes/X/Y/Z", the physical group it
    will be assigned is "Z".
    """
    function add_cad_entity_names_to_physical_groups(dim::Int64=2)
        physicals = Dict{String, Vector{Int32}}()
        for ent in gmsh.model.get_entities(dim)
            tag = ent[2]
            name = gmsh.model.get_entity_name(dim, tag)
            if name != ""
                path = split(name, "/")
                ent_name = path[end]
                if ent_name ∉ keys(physicals)
                    physicals[ent_name] = Int32[]
                end
                push!(physicals[ent_name], tag)
            end
        end
        for (name, tags) in sort(physicals)
            p = gmsh.model.add_physical_group(dim, tags)
            gmsh.model.set_physical_name(dim, p, name)
        end
        return sort!(collect(keys(physicals)))
    end
end
