function overlay_mpact_grid_hierarchy(grid::MPACTGridHierarchy, material::String)
    @info "Overlaying MPACT grid hierarchy"
    # Ensure that the material is of the form "Material: xxxx"
    if !startswith(uppercase(material), "MATERIAL: ")
        error("Material must be of the form 'Material: X'")
    end

    model_dim_tags = gmsh.model.get_entities(2)
    grid_tags = gmsh_generate_rectangular_grid(bb, x, y; material = material)
    grid_dim_tags = [(2, tag) for tag in grid_tags]
    union_of_dim_tags = vcat(model_dim_tags, grid_dim_tags)
    groups = gmsh.model.get_physical_groups()
    names = [gmsh.model.get_physical_name(grp[1], grp[2]) for grp in groups]
    material_indices = findall(x->occursin("MATERIAL", x), names)
    # material hierarchy with the grid material at the bottom.
    material_hierarchy = names[material_indices]
    push!(material_hierarchy, material)
    out_dim_tags = gmsh_group_preserving_fragment(
                        union_of_dim_tags,
                        union_of_dim_tags;
                        material_hierarchy = material_hierarchy
                   )

    return out_dim_tags    
end

#function gmsh_overlay_rectangular_grid(bb::NTuple{4, Float64},
#                                       material::String,
#                                       nx::Vector{Int64}, 
#                                       ny::Vector{Int64})
#
#    @info "Overlaying rectangular grid"
#    # Ensure that the material is of the form "MATERIAL_xxxx"
#    if (length(material) < 9) 
#        @error "material argument must of of the form 'MATERIAL_xxxxxx'" 
#    end
#    if material[1:9] != "MATERIAL_"
#        @error "material argument must of of the form 'MATERIAL_xxxxxx'" 
#    end
#
#    model_dim_tags = gmsh.model.get_entities(2)
#    grid_tags = gmsh_generate_rectangular_grid(bb, nx, ny; material = material)
#    grid_dim_tags = [(2, tag) for tag in grid_tags]
#    union_of_dim_tags = vcat(model_dim_tags, grid_dim_tags)
#    groups = gmsh.model.get_physical_groups()
#    names = [gmsh.model.get_physical_name(grp[1], grp[2]) for grp in groups]
#    material_indices = findall(x->occursin("MATERIAL", x), names)
#    # material hierarchy with the grid material at the bottom.
#    material_hierarchy = names[material_indices]
#    push!(material_hierarchy, material)
#    out_dim_tags = gmsh_group_preserving_fragment(
#                        union_of_dim_tags,
#                        union_of_dim_tags;
#                        material_hierarchy = material_hierarchy
#                   )
#
#    return out_dim_tags    
#end

                                        material::String = "MATERIAL_WATER")
    # Create the grid
    # We only need to make the smallest rectangles and group them into larger ones
    grid_tags_coords = Tuple{Int32,Float64,Float64}[]
    nlevels = length(x_full)
    x_small = x_full[nlevels]
    y_small = y_full[nlevels]
    for (yi, yv) in enumerate(y_small[1:length(y_small)-1])
        for (xi, xv) in enumerate(x_small[1:length(x_small)-1])
            tag = gmsh.model.occ.add_rectangle(xv, yv, 0, x_small[xi+1] - xv, y_small[yi+1] - yv)
            push!(grid_tags_coords, (tag, xv, yv))
        end                                      
    end
    @debug "Synchronizing model"
    gmsh.model.occ.synchronize()

    # Label the rectangles with the appropriate grid level and location
    # Create a dictionary holding all the physical group names and tags corresponding to
    # each group name.
    grid_levels_tags = Dict{String,Array{Int32,1}}()
    max_grid_digits = max(length(string(length(x_small)-1)), 
                          length(string(length(y_small)-1)))
    # Create each grid name
    for lvl in 1:nlevels
        for j in 1:length(y_full[lvl])-1
            for i in 1:length(x_full[lvl])-1
                grid_str = string("GRID_L", lvl, "_", lpad(i, max_grid_digits, "0"), "_", 
                                                      lpad(j, max_grid_digits, "0"))
                grid_levels_tags[grid_str] = Int32[]
            end
        end
    end
    # For each rectangle, find which grid level/index it belongs to.
    for (tag, x0, y0) in grid_tags_coords
        for lvl in 1:nlevels
            i = searchsortedlast(x_full[lvl], x0)
            j = searchsortedlast(y_full[lvl], y0)
            grid_str = string("GRID_L", lvl, "_", lpad(i, max_grid_digits, "0"), "_", 
                                                  lpad(j, max_grid_digits, "0"))
            push!(grid_levels_tags[grid_str], tag)
        end
    end
    @debug "Setting rectangular grid physical groups"
    for name in keys(grid_levels_tags)
        output_tag = gmsh.model.add_physical_group(2, grid_levels_tags[name])
        gmsh.model.set_physical_name(2, output_tag, name)
    end
    tags = [ tag for (tag, x0, y0) in grid_tags_coords ]
    # If there is already a physical group with this material name, then we need to erase it
    # and make a new physical group with the same name, that contains the previous entities


















