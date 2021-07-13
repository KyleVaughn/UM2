function _validate_gmsh_rectangular_grid_input(bb::Vector{T}, 
                                                x::Vector{Vector{T}}, 
                                                y::Vector{Vector{T}}) where {T <: AbstractFloat} 
    # BB has 4 values
    if length(bb) != 4
        error("Bounding box must be length 6")
    end
    # BB has valid values
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
    for xvector in enumerate(x)
        for i = 1:length(xvector)-1 
            if xvector[i] > xvector[i+1]
                error("x is not monotonically increasing.")
            end
        end
    end
    for yvector in enumerate(y)
        for i = 1:length(yvector)-1 
            if yvector[i] > yvector[i+1]
                error("y is not monotonically increasing.")
            end
        end
    end
end


function gmsh_rectangular_grid(bb::Vector{T}, 
                                x::Vector{Vector{T}}, 
                                y::Vector{Vector{T}}) where {T <: AbstractFloat} 
    @info "Generating rectangular grid in gmsh"
    _validate_gmsh_rectangular_grid_input(bb, x, y) 



end
