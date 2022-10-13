export RegularPartition2, RegPart2

export x_min, x_max, y_min, y_max, delta_x, delta_y, num_x, num_y,    
       bounding_box, width, height, children

struct RegularPartition2{P}
    id::UM_I
    name::String
    grid::RegularGrid2
    children::Matrix{P}

    function RegularPartition2(id::UM_I,
                               name::String,
                               grid::RegularGrid2,
                               children::Matrix{P}) where {P}
        grid_size = size(grid)
        # Ensure that the grid and matrix of children have the same size.
        if size(children) != grid_size
            error("The size of the children matrix must match the size of the grid.")
        end
        # Ensure that the bounds of the grid cells match the bounds of the children.
        for j in 1:grid_size[2], i in 1:grid_size[1]
            if get_box(grid, i, j) ≉ bounding_box(children[i, j])
                error("The bounds of the grid cells must match the bounds of the children.")
            end
        end
        return new{P}(id, name, grid, children)
    end

    function RegularPartition2(id::UM_I,
                               name::String,
                               grid::RegularGrid2,
                               children::Matrix{P}) where {P <: Integer}
        grid_size = size(grid)
        # Ensure that the grid and matrix of children have the same size.
        if size(children) != grid_size
            error("The size of the children matrix must match the size of the grid.")
        end
        return new{P}(id, name, grid, children)
    end

end

# -- Aliases --

const RegPart2 = RegularPartition2

# -- Methods --

children(rp::RegPart2) = rp.children
x_min(rp::RegPart2) = x_min(rp.grid)
x_max(rp::RegPart2) = x_max(rp.grid)
y_min(rp::RegPart2) = y_min(rp.grid)
y_max(rp::RegPart2) = y_max(rp.grid)
delta_x(rp::RegPart2) = delta_x(rp.grid)
delta_y(rp::RegPart2) = delta_y(rp.grid)
width(rp::RegPart2) = width(rp.grid)
height(rp::RegPart2) = height(rp.grid)
bounding_box(rp::RegPart2) = bounding_box(rp.grid)

Base.getindex(rp::RegPart2, i::I, j::I) where {I <: Integer} = rp.children[i, j]
Base.getindex(rp::RegPart2, i::I) where {I <: Integer} = rp.children[i]
Base.size(rp::RegPart2) = size(rp.children)
