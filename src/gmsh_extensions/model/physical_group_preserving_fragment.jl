"""
    physical_group_preserving_fragment(object_dtags::Vector{Tuple{Int32, Int32}},
                                       tool_dtags::Vector{Tuple{Int32, Int32}};
                                       material_hierarchy::Vector{Material} = Material[])

The equivalent to performing `gmsh.model.occ.fragment(object_dtags, tool_dtags)`, but
preserving the physical groups of the highest dimensional entities. Only highest dimensional
physical groups can be preserved due to available parent-child relationships from 
`gmsh.model.occ.fragment`

In the event that two overlapping entities have material physical groups, the optional
material hierarchy is used to choose a single material for the resultant overlapping entity.
"""
function physical_group_preserving_fragment(object_dtags::Vector{Tuple{Int32, Int32}},
                                            tool_dtags::Vector{Tuple{Int32, Int32}};
                                            material_hierarchy::Vector{Material} = Material[])
    
    # Get all physical groups and their names
    dim = max(maximum(getindex.(object_dtags, 1)), maximum(getindex.(tool_dtags, 1)))
    groups = gmsh.model.get_physical_groups(dim)
    old_tags = [gmsh.model.get_entities_for_physical_group(dtag[1], dtag[2]) for dtag in groups] 
    names = [gmsh.model.get_physical_name(dtag[1], dtag[2]) for dtag in groups]

    # Fragment
    nents = length(object_dtags) + length(tool_dtags)
    @info "... Fragmenting $nents entities"
    output_dtags, output_dtags_map = gmsh.model.occ.fragment(object_dtags, tool_dtags)

    # Update the physical groups
    # The parent at index i of input_dtags has children out_dtags_map[i]
    @info "... Updating physical groups"
    input_dtags = vcat(object_dtags, tool_dtags)
    new_tags = similar(old_tags)
    # For each physical group
    for (i, group) in enumerate(groups)
        gdim, gnum = group
        new_tags[i] = Tuple{Int32, Int32}[]
        # For each of the dim tags in the physical group
        for tag in old_tags[i]
            dtag = (gdim, tag)
            # If the dim tag was one of the entities in the fragment
            if dtag ∈ input_dtags
                # Get its children
                index = findfirst(x->x == dtag, input_dtags)
                children = output_dtags_map[index]
                # Add the children to the new physical group
                for child in children
                    if child[2] ∉ new_tags[i]
                        push!(new_tags[i], child[2])
                    end
                end
            else
                # If it wasn't in the fragment, no changes necessary.
                push!(new_tags[i], tag)
            end
        end
    end

    # Remove old groups and synchronize
    for name in names
        gmsh.model.remove_physical_name(name)
    end
    gmsh.model.occ.synchronize()

    # Process the material hierarchy, if it exists, so that each entity has one
    # or fewer materials
    if 0 < length(material_hierarchy)
        _process_material_hierarchy!(names, new_tags, material_hierarchy)
    end

    # Create the new physical groups
    for (i, group) in enumerate(groups)
        gdim, gnum = group
        ptag = gmsh.model.add_physical_group(gdim, new_tags[i])
        gmsh.model.set_physical_name(gdim, ptag, names[i])
    end

    # Apply material colors
    if 0 < length(material_hierarchy)
        color_material_physical_group_entities(material_hierarchy)
    end

    return output_dtags, output_dtags_map
end

function _process_material_hierarchy!(
        names::Vector{String},
        new_tags::Vector{Vector{Int32}},
        material_hierarchy::Vector{Material})
    material_indices = findall(x->startswith(x, "Material: "), names) 
    material_names = names[material_indices]
    material_dict = Dict{String, BitSet}()
    for (i, name) in enumerate(material_names)
        material_dict[name] = BitSet(new_tags[material_indices[i]])
    end
    # Ensure each material group is present in the hierarchy and warn now if it's not. 
    # Otherwise the error that occurs later is not as easy to figure out
    for material_name in getfield.(material_hierarchy, :name)
        if "Material: "*material_name ∉ material_names
            error("Physical groups does not contain: 'Material: $material_name'")
        end
    end
    # Order material names to the hierarchy
    material_names = ["Material: "*mat.name for mat in material_hierarchy]
    # Use the hierarchy to ensure that no entity has more than 1 material by removing
    # all shared elements from sets lower than the current set
    nmats = length(material_hierarchy)
    for i ∈ 1:nmats 
        for j ∈ i+1:nmats
            setdiff!(material_dict[material_names[j]], material_dict[material_names[i]])
        end
    end
    for (mat_index, global_index) in enumerate(material_indices)
        new_tags[global_index] = collect(material_dict[material_names[mat_index]])
    end
    return material_dict
end
