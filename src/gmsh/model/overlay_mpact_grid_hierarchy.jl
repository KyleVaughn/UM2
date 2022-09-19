export overlay_mpact_grid_hierarchy

function overlay_mpact_grid_hierarchy(grid::MPACTGridHierarchy,
                                      material_hierarchy::Vector{Material})
    @info "Overlaying MPACT grid hierarchy"
    model_dtags = gmsh.model.get_entities(2)
    int_width = 5
    lat_ctr = 0
    mod_ctr = 0
    cel_ctr = 0
    grid_tags = Int32[]
    # Sort the lattice (i, j) indices into morton order
    ilat_max, jlat_max = size(grid.lattice_grid)
    lat_zorder = Vector{u32}(undef, ilat_max * jlat_max)
    mod_zorder = u32[]
    cel_zorder = u32[]
    groups = Dict{String, Vector{Int32}}()
    for j in u32(1):u32(jlat_max), i in u32(1):u32(ilat_max)
        lat_zorder[jlat_max * (j - u32(1)) + i] = encode_morton(i - u32(1), j - u32(1))
    end
    lat_zperm = sortperm(lat_zorder)
    # Traverse the lattices in morton order
    for lat_idx in lat_zperm
        lat_ctr += 1
        lat_name = "Lattice_" * lpad(lat_ctr, int_width, '0')
        groups[lat_name] = Int32[]
        # Sort the lattice's module indices into morton order
        lat = grid.lattice_grid[lat_idx]
        imod_max, jmod_max = size(lat)
        mod_len = length(mod_zorder)
        # Resize the module z-order vector if necessary
        if imod_max * jmod_max > length(mod_zorder)
            resize!(mod_zorder, imod_max * jmod_max)
        end
        for j in u32(1):u32(jmod_max), i in u32(1):u32(imod_max)
            mod_zorder[jmod_max * (j - u32(1)) + i] = encode_morton(i - u32(1), j - u32(1))
        end
        mod_zperm = sortperm(mod_zorder)
        # Traverse the modules in morton order
        for mod_idx in mod_zperm
            mod_ctr += 1
            mod_name = "Module_" * lpad(mod_ctr, int_width, '0')
            groups[mod_name] = Int32[]
            # Sort the module's coarse cell indices into morton order
            rt_mod = lat[mod_idx]
            icell_max, jcell_max = size(rt_mod)
            icell_max -= 1 # Cells are one less than nodes
            jcell_max -= 1
            cell_len = length(cel_zorder)
            # Resize the cell z-order vector if necessary
            if icell_max * jcell_max > length(cel_zorder)
                resize!(cel_zorder, icell_max * jcell_max)
            end
            for j in u32(1):u32(jcell_max), i in u32(1):u32(icell_max)
                cel_zorder[jcell_max * (j - u32(1)) + i] = encode_morton(i - u32(1), j - u32(1))
            end
            sort!(cel_zorder)
            # Create the coarse geometry
            for z in cel_zorder
                cel_ctr += 1
                cel_name = "Cell_" * lpad(cel_ctr, int_width, '0')
                groups[cel_name] = Int32[]
                # Get the cell's (i, j) indices
                i, j = decode_morton(z)
                # Get the cell's (x, y) coordinates
                x0 = rt_mod.dims[1][i + u32(1)]
                y0 = rt_mod.dims[2][j + u32(1)]
                x1 = rt_mod.dims[1][i + u32(2)]
                y1 = rt_mod.dims[2][j + u32(2)]
                # Create the coarse cell
                tag = gmsh.model.occ.add_rectangle(x0, y0, 0.0, x1 - x0, y1 - y0)
                # Add the tag to the appropriate group
                push!(grid_tags, tag)
                push!(groups[cel_name], tag)
                push!(groups[mod_name], tag)
                push!(groups[lat_name], tag)
                
            end
        end
    end
    gmsh.model.occ.synchronize()

    # Setup material physical groups
    mat_name = "Material: " * material_hierarchy[end].name
    groups[mat_name] = grid_tags
    old_groups = gmsh.model.get_physical_groups()
    names = [gmsh.model.get_physical_name(grp[1], grp[2]) for grp in old_groups]

    # Assign new groups
    for name in keys(groups)
        safe_add_physical_group(name, [(Int32(2), t) for t in groups[name]])
    end

    output_dtags, output_dtags_map = safe_fragment(model_dtags, 
                                                   [(Int32(2), t) for t in grid_tags],
                                                   material_hierarchy = material_hierarchy)

    return output_dtags, output_dtags_map
end
