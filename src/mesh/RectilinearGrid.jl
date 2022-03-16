"""
    RectilinearGrid{T, X, Y}(bb::AABox2D{T}, xdiv::SVector{X, T}, ydiv::SVector{Y, T})

A `RectilinearGrid`, defined by a bounding box with divisions at x and y-values `xdiv`
and `ydiv`.
"""
struct RectilinearGrid2D{T, X, Y}
    bb::AABox2D{T}
    xdiv::SVector{X, T}
    ydiv::SVector{Y, T}

    function RectilinearGrid2D{T, X, Y}(bb, xdiv, ydiv) where {T, X, Y}
        if !(all(x->bb.xmin ≤ x ≤bb.xmax, xdiv) && all(y->bb.ymin ≤ y ≤bb.ymax, ydiv))
            error("Invalid RectilinearGrid")
        end
        return new{T, X, Y}(bb, xdiv, ydiv)
    end
end

RectilinearGrid2D(bb::AABox2D{T}, 
                  xdiv::SVector{X, T}, 
                  ydiv::SVector{Y, T}
                 ) where {T, X, Y} = RectilinearGrid2D{T, X, Y}(bb, xdiv, ydiv)
