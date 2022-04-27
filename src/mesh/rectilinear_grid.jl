export RectilinearGrid

struct RectilinearGrid{Dim,X,Y,Z,T}
    bb::AABox{Dim,T}
    xdiv::Vec{X,T}
    ydiv::Vec{Y,T}
    zdiv::Vec{Z,T}

    function RectilinearGrid{2,X,Y,Z,T}(bb, xdiv, ydiv) where {X,Y,Z,T}
        if X === 0
            error("Must have at least 1 X-division.")
        end
        if Y === 0
            error("Must have at least 1 Y-division.")
        end
        if Z !== 0
            error("Z-divisions not allowed in a 2D grid.")
        end
        minx = xmin(bb) 
        maxx = xmax(bb)
        miny = ymin(bb) 
        maxy = ymax(bb)
        if !(all(x->minx < x < maxx, xdiv) && 
             all(y->miny < y < maxy, ydiv))
            error("Divisions must be within the bounding box.")
        end
        if any(i->xdiv[i+1] - xdiv[i] < 0, 1:X-1) || 
           any(i->ydiv[i+1] - ydiv[i] < 0, 1:Y-1)
            error("Divisions must be monotonically increasing.")
        end
        return new{2,X,Y,Z,T}(bb, xdiv, ydiv, Vec{0,T}())
    end

    function RectilinearGrid{3,X,Y,Z,T}(bb, xdiv, ydiv, zdiv) where {X,Y,Z,T}
        if X === 0
            error("Must have at least 1 X-division.")
        end
        if Y === 0
            error("Must have at least 1 Y-division.")
        end
        if Z === 0
            error("Must have at least 1 Z-division.")
        end
        minx = xmin(bb) 
        miny = ymin(bb) 
        minz = zmin(bb)
        maxx = xmax(bb)
        maxy = ymax(bb)
        maxz = zmax(bb)
        if !(all(x->minx < x < maxx, xdiv) && 
             all(y->miny < y < maxy, ydiv) &&
             all(z->minz < z < maxz, zdiv))
            error("Divisions must be within the bounding box.")
        end
        if any(i->xdiv[i+1] - xdiv[i] < 0, 1:X-1) || 
           any(i->ydiv[i+1] - ydiv[i] < 0, 1:Y-1) ||
           any(i->zdiv[i+1] - zdiv[i] < 0, 1:Z-1)
            error("Divisions must be monotonically increasing.")
        end
        return new{3,X,Y,Z,T}(bb, xdiv, ydiv, zdiv)
    end
end

# constructors
function RectilinearGrid(bb::AABox{2,T}, xdiv::Vec{X,T}, ydiv::Vec{Y,T}) where {Dim,T,X,Y}
    return RectilinearGrid{2,X,Y,0,T}(bb, xdiv, ydiv)
end
function RectilinearGrid(bb::AABox{3,T}, xdiv::Vec{X,T}, 
                                         ydiv::Vec{Y,T},
                                         zdiv::Vec{Z,T}) where {Dim,T,X,Y,Z}
    return RectilinearGrid{3,X,Y,Z,T}(bb, xdiv, ydiv, zdiv)
end

function RectilinearGrid(bb::AABox{2,T}, xdiv, ydiv) where {T}
    X = length(xdiv)
    Y = length(ydiv)
    sxdiv = Vec{X,T}(sort(xdiv))
    sydiv = Vec{Y,T}(sort(ydiv))
    return RectilinearGrid(bb, sxdiv, sydiv)
end

function RectilinearGrid(bb::AABox{3,T}, xdiv, ydiv, zdiv) where {T}
    X = length(xdiv)
    Y = length(ydiv)
    Z = length(zdiv)
    sxdiv = Vec{X,T}(sort(xdiv))
    sydiv = Vec{Y,T}(sort(ydiv))
    szdiv = Vec{Z,T}(sort(zdiv))
    return RectilinearGrid(bb, sxdiv, sydiv, szdiv)
end

issubset(g1::RectilinearGrid, g2::RectilinearGrid) = g1.xdiv ⊆ g2.xdiv && 
                                                     g1.ydiv ⊆ g2.ydiv &&
                                                     g1.zdiv ⊆ g2.zdiv 
