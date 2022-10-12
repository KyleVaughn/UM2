export RectilinearPartition2, RectPart2

export x_min, x_max, y_min, y_max, width, height, num_x, num_y,
       bounding_box

struct RectilinearPartition2{P}
    id::UM_I
    name::String
    grid::RectilinearGrid2
    children::Matrix{P}

    function RectilinearPartition2(id::UM_I, 
                                   name::String, 
                                   grid::RectilinearGrid2,
                                   children::Matrix{P}) where {P}
        grid_size = size(grid) .- 1
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

    function RectilinearPartition2(id::UM_I, 
                                   name::String, 
                                   grid::RectilinearGrid2,
                                   children::Matrix{P}) where {P <: Integer}
        grid_size = size(grid) .- 1
        # Ensure that the grid and matrix of children have the same size.
        if size(children) != grid_size
            error("The size of the children matrix must match the size of the grid.")
        end
        return new{P}(id, name, grid, children)
    end

end

# -- Type aliases --

const RectPart2 = RectilinearPartition2

# -- Methods --

x_min(rp::RectPart2) = x_min(rp.grid)
x_max(rp::RectPart2) = x_max(rp.grid)
y_min(rp::RectPart2) = y_min(rp.grid)
y_max(rp::RectPart2) = y_max(rp.grid)
width(rp::RectPart2) = width(rp.grid)
height(rp::RectPart2) = height(rp.grid)
num_x(rp::RectPart2) = num_x(rp.grid)
num_y(rp::RectPart2) = num_y(rp.grid)
bounding_box(rp::RectPart2) = bounding_box(rp.grid)

Base.size(rp::RectPart2) = size(rp.children)
Base.getindex(rp::RectPart2, i::I, j::I) where {I <: Integer} = rp.children[i, j]
Base.getindex(rp::RectPart2, i::I) where {I <: Integer} = rp.children[i]

# Construct an index map representing the mapping of the vector of AABBs to
# a rectangular partition.
function RectilinearPartition2(bbs::Vector{AABox2{T}}) where {T}
    # Determine the number of columns by finding the number of unique x_min values.
    # Determine the number of rows by finding the number of unique y_min values.
    x_mins = T[]
    y_mins = T[]
    for bb in bbs
        bb_x = x_min(bb)
        bb_y = y_min(bb)
        # Check if the x_min value is already in the list.
        if !any(x -> abs(x - bb_x) < T(1e-4), x_mins)
            push!(x_mins, bb_x)
        end
        # Check if the y_min value is already in the list.
        if !any(y -> abs(y - bb_y) < T(1e-4), y_mins)
            push!(y_mins, bb_y)
        end
    end
    sort!(x_mins)
    sort!(y_mins)
    num_x = length(x_mins)
    num_y = length(y_mins)
    # Allocate the children matrix.
    children = fill(UM_I(0), num_x, num_y)
    # Populate the children matrix.
    for (id, bb) in enumerate(bbs)
        bb_x = x_min(bb)
        bb_y = y_min(bb)
        # Find the index of the x_min value.
        # Perturb the value slightly to avoid floating point errors.
        x_idx = searchsortedfirst(x_mins, bb_x + T(1e-4)) - 1
        if x_idx < 1 || x_idx > num_x
            error("Invalid x index.")
        end
        # Find the index of the y_min value.
        y_idx = searchsortedfirst(y_mins, bb_y + T(1e-4)) - 1
        if y_idx < 1 || y_idx > num_y
            error("Invalid y index.")
        end
        # Add the index of the bounding box to the children matrix.
        if children[x_idx, y_idx] != UM_I(0)
            error("The child matrix is not empty at index ($x_idx, $y_idx).")
        end
        children[x_idx, y_idx] = UM_I(id)
    end
    # Construct the grid.
    # The grid is constructed by using the x_min and y_min values as the
    # lower bounds of the grid cells. Therefore, we only need to add the
    # maximum x and y values to construct grid.
    max_x = maximum(x -> x_max(x), bbs)
    max_y = maximum(y -> y_max(y), bbs)
    push!(x_mins, max_x)
    push!(y_mins, max_y)
    grid = RectilinearGrid((x_mins, y_mins))
    # Ensure that the grid and matrix of children have the same size.
    grid_size = size(grid) .- 1
    children_size = size(children)
    if grid_size != children_size
        error("The size of the children matrix $children_size must match the size of the grid $grid_size")
    end
    # Ensure that the bounds of the grid cells match the bounds of the children.
    for j in 1:grid_size[2], i in 1:grid_size[1]
        if get_box(grid, i, j) ≉ bbs[children[i, j]]
            error("The bounds of the grid cells must match the bounds of the children.")
        end
    end
    return RectilinearPartition2(UM_I(0), "", grid, children)
end
