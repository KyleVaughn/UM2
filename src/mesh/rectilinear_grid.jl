export RectilinearGrid, RectilinearGrid2, RectilinearGrid2f, RectilinearGrid2d

export x_min, x_max, y_min, y_max

# RECTILINEAR GRID
# -----------------------------------------------------------------------------
#
# A D-dimensional rectilinear grid with data of type T.
#

struct RectilinearGrid{D, T}
    dims::NTuple{D, Vector{T}}
end

# -- Type aliases --

const RectilinearGrid2 = RectilinearGrid{2}
const RectilinearGrid2f = RectilinearGrid2{Float32}
const RectilinearGrid2d = RectilinearGrid2{Float64}

# -- Constructors --

function RectilinearGrid(x::Vector{T}, y::Vector{T}) where {T}
    return RectilinearGrid{2, T}(x, y)
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
