export RectilinearGrid, RectilinearGrid2, RectGrid, RectGrid2

export x_min, x_max, y_min, y_max, width, height, num_x, num_y, find_face,
       bounding_box, width, heigth, get_box

# RECTILINEAR GRID
# -----------------------------------------------------------------------------
#
# A D-dimensional rectilinear grid with data of type T.
#

struct RectilinearGrid{D}
    dims::NTuple{D, Vector{UM_F}}

    function RectilinearGrid(dims::NTuple{D, Vector{UM_F}}) where {D}
        if !all(v -> issorted(v), dims)
            throw(ArgumentError("The divisions of the grid must be sorted."))
        end
        return new{D}(dims)
    end
end

# -- Type aliases --

const RectilinearGrid2 = RectilinearGrid{2}
const RectGrid = RectilinearGrid
const RectGrid2 = RectilinearGrid{2}

# -- Constructors --

function RectilinearGrid(x::Vector{UM_F}, y::Vector{UM_F})
    return RectilinearGrid((x, y))
end

# Turn an AABB into a RegularGrid. Used in spatial partitioning.
function RectilinearGrid(aabb::AABB)
    return RectilinearGrid([x_min(aabb), x_max(aabb)], [y_min(aabb), y_max(aabb)])
end

# -- Methods --

Base.size(rg::RectilinearGrid2) = (length(rg.dims[1]) - 1, length(rg.dims[2]) - 1)

function get_box(rg::RectilinearGrid2, i::Integer, j::Integer)
    return AABox(Point2(rg.dims[1][i    ], rg.dims[2][j    ]),
                 Point2(rg.dims[1][i + 1], rg.dims[2][j + 1]))
end

function Base.issubset(g1::RectilinearGrid{D}, g2::RectilinearGrid{D}) where {D}
    for i in 1:D
        if g1.dims[i] ⊈ g2.dims[i]
            return false
        end
    end
    return true
end

# If the axis aligned bounding box is equivalent to one of the grid cells
function Base.issubset(aabb::AABox2, rg::RectilinearGrid2)
    # Just iterate over every box in the grid and check if it is approximately
    # equal to the aabb.
    # One should probably do binary search on the rg for the aabb.minima.x and
    # aabb.minima.y, then check that the next x and y are aabb.maxima,
    # but this is probably good enough for now.
    rg_size = size(rg)
    for j in 1:rg_size[2], i in 1:rg_size[1]
        if aabb ≈ get_box(rg, i, j)
            return true
        end
    end
    return false
end

x_min(rg::RectilinearGrid) = rg.dims[1][begin]
y_min(rg::RectilinearGrid) = rg.dims[2][begin]
x_max(rg::RectilinearGrid) = rg.dims[1][end]
y_max(rg::RectilinearGrid) = rg.dims[2][end]
width(rg::RectilinearGrid) = x_max(rg) - x_min(rg)
height(rg::RectilinearGrid) = y_max(rg) - y_min(rg)
num_x(rg::RectilinearGrid) = length(rg.dims[1])
num_y(rg::RectilinearGrid) = length(rg.dims[2])

bounding_box(rg::RectilinearGrid2) = AABox(Point(x_min(rg), y_min(rg)),
                                           Point(x_max(rg), y_max(rg)))

function Base.in(P::Point2{UM_F}, rg::RectilinearGrid2)
    return x_min(rg) ≤ P[1] ≤ x_max(rg) && y_min(rg) ≤ P[2] ≤ y_max(rg)
end

function find_face(P::Point2{UM_F}, rg::RectilinearGrid2)
    i = searchsortedfirst(rg.dims[1], P[1])
    j = searchsortedfirst(rg.dims[2], P[2])
    return (i, j)
end

# For easy translation
function Base.:+(rg::RectilinearGrid2, P::Point2{UM_F})
    return RectilinearGrid((rg.dims[1] .+ P[1], rg.dims[2] .+ P[2]))
end
function Base.:-(rg::RectilinearGrid2, P::Point2{UM_F})
    return RectilinearGrid((rg.dims[1] .- P[1], rg.dims[2] .- P[2]))
end

# -- Convert to RegularGrid --

function RegularGrid(rg::RectilinearGrid2)
    # Check that the grid is regular
    dx = rg.dims[1][2] - rg.dims[1][1]
    if !all(i -> rg.dims[1][i + 1] - rg.dims[1][i] ≈ dx, 1:length(rg.dims[1]) - 1)
        throw(ArgumentError("The grid is not regular."))
    end
    dy = rg.dims[2][2] - rg.dims[2][1]
    if !all(i -> rg.dims[2][i + 1] - rg.dims[2][i] ≈ dy, 1:length(rg.dims[2]) - 1)
        throw(ArgumentError("The grid is not regular."))
    end
    # Find the bottom left corner
    min_x = x_min(rg)
    min_y = y_min(rg)
    minima = Point2(min_x, min_y)
    # Find the cell size
    dx = rg.dims[1][2] - rg.dims[1][1]
    dy = rg.dims[2][2] - rg.dims[2][1]
    delta = (dx, dy)
    # Find the number of divisions
    num_x = UM_I(length(rg.dims[1]) - 1)
    num_y = UM_I(length(rg.dims[2]) - 1)
    ndivs = (num_x, num_y)
    return RegularGrid(minima, delta, ndivs)
end
