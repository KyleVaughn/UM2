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

Base.size(rg::RectilinearGrid2) = (length(rg.dims[1]), length(rg.dims[2]))

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
