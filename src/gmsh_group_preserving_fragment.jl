function process_material_hierarchy!(
        new_physical_groups::Dict{String,Array{Tuple{Int32,Int32},1}}, 
        material_hierarchy::Vector{String})
    # Get the material groups and the entities in each group
    groups = collect(keys(new_physical_groups))
    material_indices = findall(x->occursin("MATERIAL", x), groups)
    material_groups = groups[material_indices]
    material_dict = Dict{String,Array{Tuple{Int32,Int32},1}}()
    all_material_entities = Tuple{Int32,Int32}[]
    for material in material_groups
        material_dict[material] = new_physical_groups[material]
        append!(all_material_entities, new_physical_groups[material])
    end
    # Remove duplicates
    all_material_entities = collect(Set(all_material_entities))
    # For each entity with a material, ensure that it only exists in one of material groups.
    # If it exists in more than one material group, apply the material hierarchy so that the 
    # entity only has one material.
    for ent in all_material_entities
        materials = String[]
        for material in material_groups
            if ent ∈  material_dict[material]
                push!(materials, material)
            end
        end
        if 1 < length(material)
            # Apply hierarchy until the entity only has one material
            # POP from dict and material vector
        end
        @assert length(materials) == 1 "Entity $ent should only have 1 material. Materials: $materials"
    end
end

function gmsh_group_preserving_fragment(object_dim_tags:: Array{Tuple{Signed,Int32},1},
                                   tool_dim_tags:: Array{Tuple{Signed,Int32},1};
                                   material_hierarchy = String[])
    # Get all the physical groups
    old_physical_groups = Dict{String,Array{Tuple{Int32,Int32},1}}()
    groups = gmsh.model.get_physical_groups()
    names = [gmsh.model.get_physical_name(grp[1], grp[2]) for grp in groups]
    for (i, name) in enumerate(names)
        ents = gmsh.model.get_entities_for_physical_group(groups[i][1], groups[i][2])
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

    # Remove old groups and synchronize
    for name in names
        gmsh.model.remove_physical_name(name)
    end
    @info "Synchronizing model"
    gmsh.model.occ.synchronize()

    # Process the material hierarchy if it exists
    if 0 < length(material_hierarchy)
        process_material_hierarchy!(new_physical_groups, material_hierarchy)
    end

    # Create new physical groups
    for (i, name) in enumerate(names)
        dim = groups[i][1]
        tags = [dim_tag[2] for dim_tag in new_physical_groups[name]]
        ptag = gmsh.model.add_physical_group(dim, tags)
        gmsh.model.set_physical_name(dim, ptag, name)
    end

    return out_dim_tags
end
