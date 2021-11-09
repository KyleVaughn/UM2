# @code_warntype checked 2021/11/09

function gmsh_overlay_rectangular_grid(bb::NTuple{4, T},
                                       material::String,
                                       x::Vector{Vector{T}}, 
                                       y::Vector{Vector{T}}) where {T <: AbstractFloat}

    @info "Overlaying rectangular grid"
    # Ensure that the material is of the form "MATERIAL_xxxx"
    if (length(material) < 9) 
        @error "material argument must of of the form 'MATERIAL_xxxxxx'" 
    end
    if material[1:9] != "MATERIAL_"
        @error "material argument must of of the form 'MATERIAL_xxxxxx'" 
    end

    model_dim_tags = gmsh.model.get_entities(2)
    grid_tags = gmsh_rectangular_grid(bb, x, y; material = material)
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

function gmsh_overlay_rectangular_grid(bb::NTuple{4, T},
                                       material::String,
                                       nx::Vector{Int}, 
                                       ny::Vector{Int}) where {T <: AbstractFloat}

    @info "Overlaying rectangular grid"
    # Ensure that the material is of the form "MATERIAL_xxxx"
    if (length(material) < 9) 
        @error "material argument must of of the form 'MATERIAL_xxxxxx'" 
    end
    if material[1:9] != "MATERIAL_"
        @error "material argument must of of the form 'MATERIAL_xxxxxx'" 
    end

    model_dim_tags = gmsh.model.get_entities(2)
    grid_tags = gmsh_rectangular_grid(bb, nx, ny; material = material)
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
