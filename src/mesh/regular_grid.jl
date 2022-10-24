export RegularGrid, RegularGrid2, RegGrid, RegGrid2

export x_min, x_max, y_min, y_max, delta_x, delta_y, num_x, num_y,
       bounding_box, width, height

struct RegularGrid{D}
    # The bottom left corner of the grid
    minima::Point{D, UM_F}
    # The Δx, Δy, etc. of the grid
    delta::NTuple{D, UM_F}
    # The number of divisions in each dimension.
    # Must have at least 1 division to form a box.
    ncells::NTuple{D, UM_I}

    function RegularGrid(minima::Point{D, UM_F}, 
                         delta::NTuple{D, UM_F}, 
                         ncells::NTuple{D, UM_I}) where {D}
        if !all(d -> 0 < d, delta)
            throw(ArgumentError("All Δ must be positive"))
        end
        if !all(d -> 0 < d, ncells)
            throw(ArgumentError("All N must be positive"))
        end
        return new{D}(minima, delta, ncells)
    end
end

# -- Type aliases --

const RegularGrid2 = RegularGrid{2}
const RegGrid = RegularGrid
const RegGrid2 = RegularGrid2

# -- Constructors --

# Turn an AABB into a RegularGrid. Used in spatial partitioning.
function RegularGrid2(aabb::AABB2)
    return RegularGrid(aabb.minima, 
                        (width(aabb), height(aabb)), 
                        (UM_I(1), UM_I(1)))
end

# -- Methods --

x_min(rg::RegularGrid) = rg.minima[1]
y_min(rg::RegularGrid) = rg.minima[2]
x_max(rg::RegularGrid) = rg.minima[1] + rg.delta[1] * rg.ncells[1]
y_max(rg::RegularGrid) = rg.minima[2] + rg.delta[2] * rg.ncells[2]
delta_x(rg::RegularGrid) = rg.delta[1]
delta_y(rg::RegularGrid) = rg.delta[2]
num_x(rg::RegularGrid) = rg.ncells[1]
num_y(rg::RegularGrid) = rg.ncells[2]
bounding_box(rg::RegularGrid2) = AABox(rg.minima, Point(x_max(rg), y_max(rg)))
width(rg::RegularGrid2) = x_max(rg) - x_min(rg)
height(rg::RegularGrid2) = y_max(rg) - y_min(rg)

Base.size(rg::RegularGrid2) = (num_x(rg), num_y(rg))

function get_box(rg::RegularGrid2, i::Integer, j::Integer)
    if i < 1 || num_x(rg) < i
        throw(ArgumentError("i must be in [1, num_x(rg)]"))
    end
    if j < 1 || num_y(rg) < j
        throw(ArgumentError("j must be in [1, num_y(rg)]"))
    end
    return AABox(Point2(x_min(rg) + (i - 1) * delta_x(rg), 
                        y_min(rg) + (j - 1) * delta_y(rg)),
                 Point2(x_min(rg) + i * delta_x(rg), 
                        y_min(rg) + j * delta_y(rg)))
end

function Base.in(P::Point2{UM_F}, rg::RegularGrid2)
    return x_min(rg) <= P[1] <= x_max(rg) && 
           y_min(rg) <= P[2] <= y_max(rg)
end

function find_face(P::Point2{UM_F}, rg::RegularGrid2)
    i = floor(Int64, (P[1] - x_min(rg)) / delta_x(rg)) + 1
    j = floor(Int64, (P[2] - y_min(rg)) / delta_y(rg)) + 1
    return (i, j)
end
