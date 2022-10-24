export overlay_mpact_grid_hierarchy

function overlay_mpact_grid_hierarchy(sp::MPACTSpatialPartition,
                                      material_hierarchy::Vector{Material})
    @info "Overlaying MPACT grid hierarchy"
    U = MortonCode # Defined in math/morton.jl
    U1 = U(1)
    model_dtags = gmsh.model.get_entities(2)
    int_width = 5
    lat_ctr = 0
    mod_ctr = 0
    cel_ctr = 0
    grid_tags = Int32[]
    # Sort the lattice (i, j) indices into morton order
    ilat_max, jlat_max = U.(size(sp.core))
    lat_zorder = Vector{U}(undef, ilat_max * jlat_max)
    mod_zorder = U[]
    cel_zorder = U[]
    groups = Dict{String, Vector{Int32}}()
    for j in U1:jlat_max, i in U1:ilat_max
        lat_zorder[jlat_max * (j - U1) + i] = morton_encode(i - U1, j - U1)
    end
    lat_zperm = sortperm(lat_zorder)
    # Traverse the lattices in morton order
    for lat_idx in sp.core.children[lat_zperm]
        lat_ctr += 1
        lat_name = "Lattice_" * lpad(lat_ctr, int_width, '0')
        groups[lat_name] = Int32[]
        # Sort the lattice's module indices into morton order
        lat = sp.lattices[lat_idx] # A regular partition
        imod_max, jmod_max = U.(size(lat))
        mod_len = length(mod_zorder)
        # Resize the module z-order vector if necessary
        if imod_max * jmod_max > length(mod_zorder)
            resize!(mod_zorder, imod_max * jmod_max)
        end
        for j in U1:jmod_max, i in U1:imod_max
            mod_zorder[jmod_max * (j - U1) + i] = morton_encode(i - U1, j - U1)
        end
        mod_zperm = sortperm(mod_zorder)
        # Traverse the modules in morton order
        for mod_idx in lat.children[mod_zperm]
            mod_ctr += 1
            mod_name = "Module_" * lpad(mod_ctr, int_width, '0')
            groups[mod_name] = Int32[]
            # Sort the module's coarse cell indices into morton order
            rt_mod = sp.modules[mod_idx] # A rectilinear partition
            icel_max, jcel_max = U.(size(rt_mod))
            cel_len = length(cel_zorder)
            # Resize the coarse cell z-order vector if necessary
            if icel_max * jcel_max > length(cel_zorder)
                resize!(cel_zorder, icel_max * jcel_max)
            end
            for j in U1:jcel_max, i in U1:icel_max
                cel_zorder[jcel_max * (j - U1) + i] = morton_encode(i - U1, j - U1)
            end
            cel_zperm = sortperm(cel_zorder)
            # Create the coarse geometry
            for cel_idx in cel_zperm
                cel_id = rt_mod.children[cel_idx]
                cel_ctr += 1
                cel_name = "Cell_" * lpad(cel_ctr, int_width, '0')
                groups[cel_name] = Int32[]
                # Get the cell's (i, j) indices
                j = (cel_idx - 1) ÷ icel_max + 1
                i = cel_idx - (j - 1) * icel_max 
                # Get the cell's (x, y) coordinates
                aabb = get_box(rt_mod.grid, i, j)
                x0 = x_min(aabb)
                y0 = y_min(aabb)
                x1 = x_max(aabb)
                y1 = y_max(aabb)
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
