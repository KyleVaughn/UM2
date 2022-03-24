function overlay_mpact_grid_hierarchy(grid::MPACTGridHierarchy, material::Material)
    @info "Overlaying MPACT grid hierarchy"
    model_dtags = gmsh.model.get_entities(2)
    grid_tags_coords = Tuple{Int32, Float64, Float64}[]

    xvals = collect(grid.coarse_grid.xdiv)
    prepend!(xvals, [grid.coarse_grid.bb.xmin])
    push!(xvals, grid.coarse_grid.bb.xmax)

    yvals = collect(grid.coarse_grid.ydiv)
    prepend!(yvals, [grid.coarse_grid.bb.ymin])
    push!(yvals, grid.coarse_grid.bb.ymax)

    for (iy, y) in enumerate(yvals[1:end-1])
        for (ix, x) in enumerate(xvals[1:end-1])
            tag = gmsh.model.occ.add_rectangle(x, y, 0, xvals[ix+1] - x, yvals[iy+1] - y)
            push!(grid_tags_coords, (tag, x, y))
        end                                      
    end
    gmsh.model.occ.synchronize()

    # Label the rectangles with the appropriate physical groups 
    # Create a dictionary holding all the physical group names and tags
    groups = Dict{String, Vector{Int32}}()
    max_grid_digits = max(length(string(length(xvals)-1)), 
                          length(string(length(yvals)-1)))

    # Create each grid name
    # Lattices
    for ix in 1:length(grid.lattice_grid.xdiv) + 1
        for iy in 1:length(grid.lattice_grid.ydiv) + 1
            grid_str = string("Lattice (", lpad(ix, max_grid_digits, "0"), ", ", 
                                           lpad(iy, max_grid_digits, "0"), ")")
            groups[grid_str] = Int32[]
        end
    end

    # Modules
    for ix in 1:length(grid.module_grid.xdiv) + 1
        for iy in 1:length(grid.module_grid.ydiv) + 1
            grid_str = string("Module (", lpad(ix, max_grid_digits, "0"), ", ", 
                                          lpad(iy, max_grid_digits, "0"), ")")
            groups[grid_str] = Int32[]
        end
    end

    # Coarse Cells
    for ix in 1:length(grid.coarse_grid.xdiv) + 1
        for iy in 1:length(grid.coarse_grid.ydiv) + 1
            grid_str = string("Coarse Cell (", lpad(ix, max_grid_digits, "0"), ", ", 
                                               lpad(iy, max_grid_digits, "0"), ")")
            groups[grid_str] = Int32[]
        end
    end

    # For each rectangle, find which grid/index it belongs to
    # Lattices
    xvals = collect(grid.lattice_grid.xdiv)        
    push!(xvals, grid.lattice_grid.bb.xmax)
 
    yvals = collect(grid.lattice_grid.ydiv)
    push!(yvals, grid.lattice_grid.bb.ymax)

    for (tag, x, y) in grid_tags_coords
        ix = searchsortedfirst(xvals, x)
        iy = searchsortedfirst(yvals, y)
        grid_str = string("Lattice (", lpad(ix, max_grid_digits, "0"), ", ", 
                                       lpad(iy, max_grid_digits, "0"), ")")
        push!(groups[grid_str], tag)
    end














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

    # Account for material already in model, and erase an add.
    return out_dim_tags    
end










    @debug "Setting rectangular grid physical groups"
    for name in keys(grid_levels_tags)
        output_tag = gmsh.model.add_physical_group(2, grid_levels_tags[name])
        gmsh.model.set_physical_name(2, output_tag, name)
    end
    tags = [ tag for (tag, x0, y0) in grid_tags_coords ]
    # If there is already a physical group with this material name, then we need to erase it
    # and make a new physical group with the same name, that contains the previous entities
