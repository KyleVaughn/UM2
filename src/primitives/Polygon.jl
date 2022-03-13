"""
    Polygon(SVector{N, Point{Dim, T}})
    Polygon(p₁::Point{Dim, T}}, p₂::Point{Dim, T}}, ...)

Construct a `Polygon` with `N` counter-clockwise oriented vertices in 
`Dim`-dimensional space. Several aliases exist for convenience, e.g. Triangle (`N`=3),
Quadrilateral (`N`=4), etc.
"""
struct Polygon{N, Dim, T} <:Face{Dim, 1, T}
    points::SVector{N, Point{Dim, T}}
end

# Aliases for convenience
const Polygon2D       = Polygon{N,2} where {N}
const Polygon3D       = Polygon{N,3} where {N}
const Triangle        = Polygon{3}
const Quadrilateral   = Polygon{4}
const Hexagon         = Polygon{6}
const Triangle2D      = Polygon{3,2}
const Quadrilateral2D = Polygon{4,2}
const Triangle3D      = Polygon{3,3}
const Quadrilateral3D = Polygon{4,3}

Base.@propagate_inbounds function Base.getindex(poly::Polygon, i::Integer)
    getfield(poly, :points)[i]
end

function Polygon{N}(v::SVector{N, Point{Dim, T}}) where {N, Dim, T}
    return Polygon{N, Dim, T}(v)
end
Polygon{N}(x...) where {N} = Polygon(SVector(x))
Polygon(x...) = Polygon(SVector(x))
