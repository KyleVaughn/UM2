function _validate_gmsh_rectangular_grid_input(bb::NTuple{4, T}, 
                                                x::Vector{Vector{T}}, 
                                                y::Vector{Vector{T}}) where {T <: AbstractFloat} 
    # bb has valid values
    xmin, xmax, ymin, ymax = bb
    dx = xmax - xmin
    dy = ymax - ymin
    if dx <= 0 || dy <= 0
        error("Invalid bounding box. Must be of the form [xmin, xmax, ymin, ymax]")
    end
    # x and y have same number of levels
    if length(x) != length(y)
        error("x and y must have the same length") 
    end
    # monotonic x, y
    for xvector in x
        for i = 1:length(xvector)-1 
            if xvector[i] > xvector[i+1]
                error("x is not monotonically increasing.")
            end
        end
    end
    for yvector in y
        for i = 1:length(yvector)-1 
            if yvector[i] > yvector[i+1]
                error("y is not monotonically increasing.")
            end
        end
    end
    # x, y in bb
    for xvector in x
        for i = 1:length(xvector)
            if !(xmin <= xvector[i] <= xmax)
                errval = xvector[i]
                error("x value ($errval) is not in the bounding box.")
            end
        end
    end
    for yvector in y
        for i = 1:length(yvector)
            if !(ymin <= yvector[i] <= ymax)
                errval = yvector[i]
                error("y value ($errval) is not in the bounding box.")
            end
        end
    end

    # Add bb limits to x, y and propagate all low level (coarse) divisions to higher levels (fine)
    x_full = deepcopy(x)
    y_full = deepcopy(y)
    append!(x_full[1], [xmin, xmax])
    append!(y_full[1], [ymin, ymax])
    for i = 2:length(x_full)
        append!(x_full[i], x_full[i-1])
        append!(y_full[i], y_full[i-1])
    end
    for i = 1:length(x_full)
        x_full[i] = sort!(collect(Set(x_full[i])))
        y_full[i] = sort!(collect(Set(y_full[i])))
    end

    # Check that all modular geometry sizes are the same (nlevels - 1)
    nlevels = length(x_full)
    if 1 < nlevels
        dx = x_full[nlevels-1][2] - x_full[nlevels-1][1]
        dy = y_full[nlevels-1][2] - y_full[nlevels-1][1]
        for i = 2:length(x_full[nlevels-1])-1
            if !(x_full[nlevels-1][i+1] - x_full[nlevels-1][i] ≈ dx)
                error(string("Grid level $nlevels must have equal x-divisions so that all ",
                             "modular geometry have the same size!"))
            end
        end
        for i = 2:length(y_full[nlevels-1])-1
            if !(y_full[nlevels-1][i+1] - y_full[nlevels-1][i] ≈ dy)
                error(string("Grid level $nlevels must have equal y-divisions so that all ",
                             "modular geometry have the same size!"))
            end
        end
    end
    return x_full, y_full
end

function gmsh_rectangular_grid(bb::NTuple{4, T}, 
                                x::Vector{Vector{T}}, 
                                y::Vector{Vector{T}}) where {T <: AbstractFloat} 
    @info "Generating rectangular grid in gmsh"
    x_full, y_full = _validate_gmsh_rectangular_grid_input(bb, x, y) 

    # Create the grid
    # We only need to make the smallest rectangles and group them into larger ones
    grid_tags_coords = Tuple{Int64,Float64,Float64}[]
    nlevels = length(x_full)
    x_small = x_full[nlevels]
    y_small = y_full[nlevels]
    for (yi, yv) in enumerate(y_small[1:length(y_small)-1])
        for (xi, xv) in enumerate(x_small[1:length(x_small)-1])
            tag = gmsh.model.occ.addRectangle(xv, yv, 0, x_small[xi+1] - xv, y_small[yi+1] - yv)
            push!(grid_tags_coords, (tag, xv, yv))
        end                                      
    end
    @info "Synchronizing model"
    gmsh.model.occ.synchronize()

    # Label the rectangles with the appropriate grid level and location
    # Create a dictionary holding all the physical group names and tag IDs corresponding to
    # each group name.



end
