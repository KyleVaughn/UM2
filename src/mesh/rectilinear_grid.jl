export RectilinearGrid
export xvals, yvals, zvals, xmin, ymin, zmin, xmax, ymax, zmax

struct RectilinearGrid{X,Y,Z,T}
    x::Vec{X,T}
    y::Vec{Y,T}
    z::Vec{Z,T}

    function RectilinearGrid{X,Y,Z,T}(x, y, z) where {X,Y,Z,T}
        if X < 2
            error("Must have at least 2 X-divisions.")
        end
        if Y < 2
            error("Must have at least 2 Y-divisions.")
        end
        if Z !== 0
            if Z < 2
                error("Must have at least 2 Z-divisions.")
            end
            if any(i->z[i+1] - z[i] < 0, 1:Z-1)
                error("Divisions must be monotonically increasing.")
            end
        end
        if any(i->x[i+1] - x[i] < 0, 1:X-1) || 
           any(i->y[i+1] - y[i] < 0, 1:Y-1)
            error("Divisions must be monotonically increasing.")
        end
        return new{X,Y,Z,T}(x, y, z)
    end
end

# constructors
function RectilinearGrid(x::Vec{X,T}, y::Vec{Y,T}) where {X,Y,T}
    return RectilinearGrid{X,Y,0,T}(x, y, Vec{0,T}())
end

function RectilinearGrid(x::Vec{X,T}, y::Vec{Y,T}, z::Vec{Z,T}) where {X,Y,Z,T}
    return RectilinearGrid{X,Y,Z,T}(x, y, z)
end

function RectilinearGrid(x, y)
    X = length(x)
    Y = length(y)
    sx = Vec{X}(x)
    sy = Vec{Y}(y)
    return RectilinearGrid(sx, sy)
end

function RectilinearGrid(x, y, z)
    X = length(x)
    Y = length(y)
    Z = length(z)
    sx = Vec{X}(x)
    sy = Vec{Y}(y)
    sz = Vec{Z}(z)
    return RectilinearGrid(sx, sy, sz)
end

Base.issubset(g1::RectilinearGrid, g2::RectilinearGrid) = g1.x ⊆ g2.x && 
                                                          g1.y ⊆ g2.y &&
                                                          g1.z ⊆ g2.z 
xvals(rg::RectilinearGrid) = rg.x
yvals(rg::RectilinearGrid) = rg.y
zvals(rg::RectilinearGrid) = rg.z
xmin(rg::RectilinearGrid) = rg.x[1]
ymin(rg::RectilinearGrid) = rg.y[1]
zmin(rg::RectilinearGrid) = rg.z[1]
xmax(rg::RectilinearGrid) = rg.x[end]
ymax(rg::RectilinearGrid) = rg.y[end]
zmax(rg::RectilinearGrid) = rg.z[end]
