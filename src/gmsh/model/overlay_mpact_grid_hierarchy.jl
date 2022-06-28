export overlay_mpact_grid_hierarchy

function overlay_mpact_grid_hierarchy(grid::MPACTGridHierarchy,
                                      material_hierarchy::Vector{Material})
    @info "Overlaying MPACT grid hierarchy"
    # Generate all of the rectangles the make up the coarse grid, and save
    # their (x,y) origin coordinates, so we can group them into their appropriate
    # modules and lattices later
    model_dtags = gmsh.model.get_entities(2)
    lg_x = grid.lattice_grid.x
    lg_y = grid.lattice_grid.y
    lg_nx = length(lg_x)
    lg_ny = length(lg_y)
    mg_x = grid.module_grid.x
    mg_y = grid.module_grid.y
    mg_nx = length(mg_x)
    mg_ny = length(mg_y)
    cg_x = grid.coarse_grid.x
    cg_y = grid.coarse_grid.y
    cg_nx = length(cg_x)
    cg_ny = length(cg_y)
    grid_tags_coords = Vector{Tuple{Int32, Float64, Float64}}(undef,
                                                              (cg_nx - 1) * (cg_ny - 1))
    gtctr = 1
    for iy in 1:(cg_ny - 1)
        y = cg_y[iy]
        for ix in 1:(cg_nx - 1)
            x = cg_x[ix]
            tag = gmsh.model.occ.add_rectangle(x, y, 0, cg_x[ix + 1] - x, cg_y[iy + 1] - y)
            grid_tags_coords[gtctr] = (tag, x, y)
            gtctr += 1
        end
    end
    gmsh.model.occ.synchronize()
    grid_dtags = [(Int32(2), gtc[1]) for gtc in grid_tags_coords]

    # Label the rectangles with the appropriate physical groups 
    # Create a dictionary holding all the physical group names and tags
    groups = Dict{String, Vector{Int32}}()
    max_grid_digits = max(length(string(cg_nx - 1)),
                          length(string(cg_ny - 1)))

    # Create each grid name
    # Lattices
    for ix in 1:(lg_nx - 1)
        for iy in 1:(lg_ny - 1)
            grid_str = string("Lattice (", lpad(ix, max_grid_digits, "0"), ", ",
                              lpad(iy, max_grid_digits, "0"), ")")
            groups[grid_str] = Int32[]
        end
    end

    # Modules
    for ix in 1:(mg_nx - 1)
        for iy in 1:(mg_ny - 1)
            grid_str = string("Module (", lpad(ix, max_grid_digits, "0"), ", ",
                              lpad(iy, max_grid_digits, "0"), ")")
            groups[grid_str] = Int32[]
        end
    end

    # Coarse Cells
    for ix in 1:(cg_nx - 1)
        for iy in 1:(cg_ny - 1)
            grid_str = string("Coarse Cell (", lpad(ix, max_grid_digits, "0"), ", ",
                              lpad(iy, max_grid_digits, "0"), ")")
            groups[grid_str] = Int32[]
        end
    end

    # For each rectangle, find which lattice/module/coarse cell it belongs to
    # Lattices
    for (tag, x, y) in grid_tags_coords
        ix = searchsortedfirst(lg_x, x)
        if x == lg_x[ix]
            ix += 1
        end
        iy = searchsortedfirst(lg_y, y)
        if y == lg_y[iy]
            iy += 1
        end
        grid_str = string("Lattice (", lpad(ix - 1, max_grid_digits, "0"), ", ",
                          lpad(iy - 1, max_grid_digits, "0"), ")")
        push!(groups[grid_str], tag)
    end

    # Modules
    for (tag, x, y) in grid_tags_coords
        ix = searchsortedfirst(mg_x, x)
        if x == mg_x[ix]
            ix += 1
        end
        iy = searchsortedfirst(mg_y, y)
        if y == mg_y[iy]
            iy += 1
        end
        grid_str = string("Module (", lpad(ix - 1, max_grid_digits, "0"), ", ",
                          lpad(iy - 1, max_grid_digits, "0"), ")")
        push!(groups[grid_str], tag)
    end

    # Coarse cells
    for (tag, x, y) in grid_tags_coords
        ix = searchsortedfirst(cg_x, x)
        if x == cg_x[ix]
            ix += 1
        end
        iy = searchsortedfirst(cg_y, y)
        if y == cg_y[iy]
            iy += 1
        end
        grid_str = string("Coarse Cell (", lpad(ix - 1, max_grid_digits, "0"), ", ",
                          lpad(iy - 1, max_grid_digits, "0"), ")")
        push!(groups[grid_str], tag)
    end

    # Setup material physical group
    mat_name = "Material: " * material_hierarchy[end].name
    groups[mat_name] = [dt[2] for dt in grid_dtags]
    old_groups = gmsh.model.get_physical_groups()
    names = [gmsh.model.get_physical_name(grp[1], grp[2]) for grp in old_groups]

    # Assign groups
    for name in keys(groups)
        safe_add_physical_group(name, [(Int32(2), t) for t in groups[name]])
    end

    # material hierarchy with the grid material at the bottom.
    output_dtags, output_dtags_map = safe_fragment(model_dtags, grid_dtags,
                                                   material_hierarchy = material_hierarchy)

    # Account for material already in model, and erase an add.
    return output_dtags, output_dtags_map
end
