export overlay_mpact_grid_hierarchy

function overlay_mpact_grid_hierarchy(grid::MPACTGridHierarchy, 
                                      material_hierarchy::Vector{Material})
    @info "Overlaying MPACT grid hierarchy"
    # Generate all of the rectangles the make up the coarse grid, and save
    # their (x,y) origin coordinates, so we can group them into their appropriate
    # modules and lattices later
    model_dtags = gmsh.model.get_entities(2)
    xvals = collect(grid.coarse_grid.x)
    yvals = collect(grid.coarse_grid.y)
    grid_tags_coords = Vector{Tuple{Int32, Float64, Float64}}(
                            undef, (length(xvals)-1)*(length(yvals)-1))
    gtctr = 1
    for (iy, y) in enumerate(yvals[1:end-1])
        for (ix, x) in enumerate(xvals[1:end-1])
            tag = gmsh.model.occ.add_rectangle(x, y, 0, xvals[ix+1] - x, yvals[iy+1] - y)
            grid_tags_coords[gtctr] = (tag, x, y)
            gtctr += 1
        end                                      
    end
    gmsh.model.occ.synchronize()
    grid_dtags = [(Int32(2), gtc[1]) for gtc in grid_tags_coords]

    # Label the rectangles with the appropriate physical groups 
    # Create a dictionary holding all the physical group names and tags
    groups = Dict{String, Vector{Int32}}()
    max_grid_digits = max(length(string(length(xvals)-1)), 
                          length(string(length(yvals)-1)))

    # Create each grid name
    # Lattices
    for ix in 1:length(grid.lattice_grid.x) - 1
        for iy in 1:length(grid.lattice_grid.y) - 1
            grid_str = string("Lattice (", lpad(ix, max_grid_digits, "0"), ", ", 
                                           lpad(iy, max_grid_digits, "0"), ")")
            groups[grid_str] = Int32[]
        end
    end

    # Modules
    for ix in 1:length(grid.module_grid.x) - 1
        for iy in 1:length(grid.module_grid.y) - 1
            grid_str = string("Module (", lpad(ix, max_grid_digits, "0"), ", ", 
                                          lpad(iy, max_grid_digits, "0"), ")")
            groups[grid_str] = Int32[]
        end
    end

    # Coarse Cells
    for ix in 1:length(grid.coarse_grid.x) - 1
        for iy in 1:length(grid.coarse_grid.y) - 1
            grid_str = string("Coarse Cell (", lpad(ix, max_grid_digits, "0"), ", ", 
                                               lpad(iy, max_grid_digits, "0"), ")")
            groups[grid_str] = Int32[]
        end
    end

    # For each rectangle, find which lattice/module/coarse cell it belongs to
    # Lattices
    xvals = collect(grid.lattice_grid.x[2:end])        
    yvals = collect(grid.lattice_grid.y[2:end])
    for (tag, x, y) in grid_tags_coords
        ix = searchsortedfirst(xvals, x)
        if xvals[ix] == x
            ix += 1
        end
        iy = searchsortedfirst(yvals, y)
        if xvals[iy] == y
            iy += 1
        end
        grid_str = string("Lattice (", lpad(ix, max_grid_digits, "0"), ", ", 
                                       lpad(iy, max_grid_digits, "0"), ")")
        push!(groups[grid_str], tag)
    end

    # Modules
    xvals = collect(grid.module_grid.x[2:end])        
    yvals = collect(grid.module_grid.y[2:end])
    for (tag, x, y) in grid_tags_coords
        ix = searchsortedfirst(xvals, x)
        if xvals[ix] == x
            ix += 1
        end
        iy = searchsortedfirst(yvals, y)
        if xvals[iy] == y
            iy += 1
        end
        grid_str = string("Module (", lpad(ix, max_grid_digits, "0"), ", ", 
                                      lpad(iy, max_grid_digits, "0"), ")")
        push!(groups[grid_str], tag)
    end

    # Coarse cells
    xvals = collect(grid.coarse_grid.x[2:end])        
    yvals = collect(grid.coarse_grid.y[2:end])
    for (tag, x, y) in grid_tags_coords
        ix = searchsortedfirst(xvals, x)
        if xvals[ix] == x
            ix += 1
        end
        iy = searchsortedfirst(yvals, y)
        if xvals[iy] == y
            iy += 1
        end
        grid_str = string("Coarse Cell (", lpad(ix, max_grid_digits, "0"), ", ", 
                                           lpad(iy, max_grid_digits, "0"), ")")
        push!(groups[grid_str], tag)
    end

    # Setup material physical group
    mat_name = "Material: "*material_hierarchy[end].name
    groups[mat_name] = [dt[2] for dt in grid_dtags] 
    old_groups = gmsh.model.get_physical_groups()
    names = [gmsh.model.get_physical_name(grp[1], grp[2]) for grp in old_groups]

    # Assign groups
    for name in keys(groups)
        safe_add_physical_group(name, [(Int32(2),t) for t in groups[name]])
    end

    # material hierarchy with the grid material at the bottom.
    output_dtags, output_dtags_map = safe_fragment(
                        model_dtags, grid_dtags,
                        material_hierarchy = material_hierarchy
                   )

    # Account for material already in model, and erase an add.
    return output_dtags, output_dtags_map 
end
