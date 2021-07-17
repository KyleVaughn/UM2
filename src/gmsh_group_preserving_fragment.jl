function group_preserving_fragment(object_dim_tags::Vector{Int},
                                   tool_dim_tags::Vector{Int};
                                   material_hierarchy = String[])
    # Get all the physical groups
    old_physical_groups = Dict{String,Array{Tuple{Int32,Int32},1}}()
    groups = gmsh.model.getPhysicalGroups()
    names = [gmsh.model.getPhysicalName(grp[1], grp[2]) for grp in groups]
    for (i, name) in enumerate(names)
        ents = gmsh.model.getEntitiesForPhysicalGroup(groups[i][1], groups[i][2])
        dim = groups[i][1]
        old_physical_groups[name] = [(dim, ent) for ent in ents]
    end

    # Fragment
    nents = length(object_dim_tags) + length(tool_dim_tags)
    @info "Fragmenting $nents entities"
    out_dim_tags, out_dim_tags_map = gmsh.model.occ.fragment(object_dim_tags, tool_dim_tags)

    # Create a dictionary of new physical groups using the parent child relationship
    # between input_dim_tags and out_dim_tags_map. The parent at index i of input_dim_tags 
    # has children out_dim_tags_map[i]
    new_physical_groups = Dict{String,Array{Tuple{Int32,Int32},1}}()
    input_dim_tags = vcat(object_dim_tags, tool_dim_tags)
    # For each physical group
    for name in names
        new_physical_groups[name] = Tuple{Int32,Int32}[]
        # For each of the dim tags in the physical group
        for dim_tag in old_physical_groups[name]
            # If the dim_tag was one of the entities in the fragment
            if dim_tag ∈  input_dim_tags
                # Get its children
                index = findfirst(x->x == dim_tag, input_dim_tags)
                children = out_dim_tags_map[index]
                # Add the children to the new physical group
                for child in children
                    if child ∉  new_physical_groups[name]
                        push!(new_physical_groups[name], child)
                    end
                end
            else
                # If it wasn't in the fragment, no changes necessary.
                push!(new_physical_groups[name], dim_tag)
            end
        end
    end





    # Check that no two entities has more than one MATERIAL_X tag
    # Make sure each material actually exists



end
