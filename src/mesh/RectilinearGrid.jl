"""
    RectilinearGrid2D{T, X, Y}(bb::AABox2D{T}, xdiv::SVector{X, T}, ydiv::SVector{Y, T})

A 2D rectilinear grid, defined by a bounding box with divisions at x and y-values 
`xdiv` and `ydiv`.
"""
struct RectilinearGrid2D{T, X, Y}
    bb::AABox2D{T}
    xdiv::SVector{X, T}
    ydiv::SVector{Y, T}

    function RectilinearGrid2D{T, X, Y}(bb, xdiv, ydiv) where {T, X, Y}
        if !(all(x->bb.xmin < x < bb.xmax, xdiv) && 
             all(y->bb.ymin < y < bb.ymax, ydiv))
            error("Invalid RectilinearGrid")
        end
        if any(i->xdiv[i+1] - xdiv[i] < 0, 1:X-1) || 
           any(i->ydiv[i+1] - ydiv[i] < 0, 1:Y-1)
            error("Divisions must be monotonically increasing")
        end
        return new{T, X, Y}(bb, xdiv, ydiv)
    end
end

function RectilinearGrid2D(bb::AABox2D{T}, 
                           xdiv::SVector{X, T}, 
                           ydiv::SVector{Y, T}
                          ) where {T, X, Y}
    return RectilinearGrid2D{T, X, Y}(bb, xdiv, ydiv)
end

function RectilinearGrid2D(bb::AABox2D{T}, 
                           xdiv::NTuple{X, T}, 
                           ydiv::NTuple{Y, T}
                          ) where {T, X, Y}
    return RectilinearGrid2D{T, X, Y}(bb, SVector(xdiv), SVector(ydiv))
end

function RectilinearGrid2D(bb::AABox2D{T}, xdiv::Vector{T}, 
                                           ydiv::Vector{T}) where {T}
    X = length(xdiv)
    Y = length(ydiv)
    sxdiv = SVector{X, T}(sort(xdiv))
    sydiv = SVector{Y, T}(sort(ydiv))
    return RectilinearGrid2D{T, X, Y}(bb, sxdiv, sydiv)
end

RectilinearGrid(bb, xdiv, ydiv) = RectilinearGrid2D(bb, xdiv, ydiv)

issubset(g1::RectilinearGrid2D, g2::RectilinearGrid2D) = g1.xdiv ⊆ g2.xdiv && 
                                                         g1.ydiv ⊆ g2.ydiv
