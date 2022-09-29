export RectilinearGrid, RectilinearGrid2, RectilinearGrid2f, RectilinearGrid2d

export x_min, x_max, y_min, y_max, delta_x, delta_y, num_x, num_y, find_face

# RECTILINEAR GRID
# -----------------------------------------------------------------------------
#
# A D-dimensional rectilinear grid with data of type T.
#

struct RectilinearGrid{D, T}
    dims::NTuple{D, Vector{T}}

    function RectilinearGrid(dims::NTuple{D, Vector{T}}) where {D, T}
        if !all(v -> issorted(v), dims)
            throw(ArgumentError("The divisions of the grid must be sorted."))
        end
        return new{D, T}(dims)
    end
end

# -- Type aliases --

const RectilinearGrid2 = RectilinearGrid{2}
const RectilinearGrid2f = RectilinearGrid2{Float32}
const RectilinearGrid2d = RectilinearGrid2{Float64}

# -- Constructors --

function RectilinearGrid(x::Vector{T}, y::Vector{T}) where {T}
    return RectilinearGrid((x, y))
end

# -- Methods --

Base.size(rg::RectilinearGrid2) = (length(rg.dims[1]), length(rg.dims[2]))

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
delta_x(rg::RectilinearGrid) = x_max(rg) - x_min(rg)
delta_y(rg::RectilinearGrid) = y_max(rg) - y_min(rg)
num_x(rg::RectilinearGrid) = length(rg.dims[1])
num_y(rg::RectilinearGrid) = length(rg.dims[2])

function Base.in(P::Point2{T}, rg::RectilinearGrid2{T}) where {T}
    return x_min(rg) ≤ P.x ≤ x_max(rg) && y_min(rg) ≤ P.y ≤ y_max(rg)
end

function find_face(P::Point2{T}, rg::RectilinearGrid2{T}) where {T}
    i = searchsortedfirst(rg.dims[1], P[1])
    j = searchsortedfirst(rg.dims[2], P[2])
    return (i, j)
end

# For easy translation
function Base.:+(rg::RectilinearGrid2{T}, P::Point2{T}) where {T}
    return RectilinearGrid((rg.dims[1] .+ P[1], rg.dims[2] .+ P[2]))
end
function Base.:-(rg::RectilinearGrid2{T}, P::Point2{T}) where {T}
    return RectilinearGrid((rg.dims[1] .- P[1], rg.dims[2] .- P[2]))
end
