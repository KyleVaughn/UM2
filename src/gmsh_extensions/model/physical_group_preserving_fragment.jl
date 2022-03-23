function physical_group_preserving_fragment(object_dtags::Vector{Tuple{Int32, Int32}},
                                            tool_dtags::Vector{Tuple{Int32, Int32}};
                                            material_hierarchy::Vector{String} = String[])
    # Only works on the highest dim entities, since that's all we get in the output of fragment
    
    # Get all the physical groups
    max_dim = max(maxi
    groups = gmsh.model.get_physical_groups()
    old_tags =  [gmsh.model.get_entities_for_physical_group(dtag[1], dtag[2]) for dtag in groups] 
    names = [gmsh.model.get_physical_name(dtag[1], dtag[2]) for dtag in groups]

    # Fragment
    nents = length(object_dtags) + length(tool_dtags)
    @info "Fragmenting $nents entities"
    output_dtags, output_dtags_map = gmsh.model.occ.fragment(object_dtags, tool_dtags)

    # Create a dictionary of new physical groups using the parent child relationship
    # between input_dim_tags and out_dim_tags_map. 
    # The parent at index i of input_dim_tags has children out_dim_tags_map[i]
    @info "Updating physical groups"
    input_dtags = vcat(object_dtags, tool_dtags)
    new_tags = similar(old_tags)
    # For each physical group
    for (i, group) in enumerate(groups)
        dim, grp_num = group
        new_tags[i] = Tuple{Int32, Int32}[]
        # For each of the dim tags in the physical group
        for tag in old_tags[i]
            dtag = (dim, tag)
            # If the dim tag was one of the entities in the fragment
            if dtag ∈ input_dtags
                # Get its children
                index = findfirst(x->x == dtag, input_dtags)
                children = output_dtags_map[index]
                # Add the children to the new physical group
                for child in children
                    if child ∉ new_tags[i]
                        push!(new_tags[i], child)
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
    gmsh.model.occ.synchronize()

    # Process the material hierarchy, if it exists, so that each entity has one
    # or less material physical groups
    if 0 < length(material_hierarchy)
        _process_material_hierarchy!(new_physical_groups, material_hierarchy)
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

function _process_material_hierarchy!(
        new_physical_groups::Dict{String, Vector{Tuple{Int32, Int32}}}, 
        material_hierarchy::Vector{String})
    # Get the material groups and the entities in each group
    groups = collect(keys(new_physical_groups))
    material_indices = findall(x->occursin("MATERIAL", uppercase(x)), groups)
    material_groups = groups[material_indices]
    # Ensure each material group is present in the hierarchy and warn now if it's not. 
    # Otherwise the error that occurs later is not as easy to figure out
    for material in material_groups
        if material ∉ material_hierarchy 
            error("Material_hierarchy does not contain: '$material'")
        end
    end
    material_dict = Dict{String, Vector{Tuple{Int32, Int32}}}()
    all_material_entities = Tuple{Int32,Int32}[]
    for material in material_groups
        # Note that this is assignment by reference. 
        # Changes to material_dict are reflected in new_physical_groups
        material_dict[material] = new_physical_groups[material]
        append!(all_material_entities, new_physical_groups[material])
    end
    # Remove duplicates
    unique!(all_material_entities)
    # For each entity with a material, ensure that it only exists in one of material 
    # groups. If it exists in more than one material group, apply the material hierarchy 
    # so that the entity only has one material.
    numerical_material_hierarchy = Dict{String,Int32}()
    i = 1
    for material in material_hierarchy
        numerical_material_hierarchy[material] = i
        i += 1
    end
    for ent in all_material_entities
        materials = String[]
        for material in material_groups
            if ent ∈  material_dict[material]
                push!(materials, material)
            end
        end
        if 1 < length(materials)
            # Get the highest priority material
            mat_num = minimum([numerical_material_hierarchy[mat] for mat in materials])
            priority_mat = material_hierarchy[mat_num]
            # Pop ent from all other materials in dict
            deleteat!(materials, materials .== priority_mat)
            for material in materials
                deleteat!(material_dict[material], 
                          findfirst(x-> x == ent, material_dict[material]))
            end
        end
    end
end
