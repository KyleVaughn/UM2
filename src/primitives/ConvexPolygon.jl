"""
    ConvexPolygon(SVector{N, Point{Dim, T}})
    ConvexPolygon(p₁::Point{Dim, T}}, p₂::Point{Dim, T}}, ...)

Construct a `ConvexPolygon` with `N` counter-clockwise oriented vertices in 
`Dim`-dimensional space. Several aliases exist for convenience, e.g. Triangle (`N`=3),
Quadrilateral (`N`=4), etc.
"""
struct ConvexPolygon{N, Dim, T} <:Face{Dim, 1, T}
    points::SVector{N, Point{Dim, T}}
end

# Aliases for convenience
const Triangle        = ConvexPolygon{3}
const Quadrilateral   = ConvexPolygon{4}
const Hexagon         = ConvexPolygon{6}
const Triangle2D      = ConvexPolygon{3,2}
const Quadrilateral2D = ConvexPolygon{4,2}
const Triangle3D      = ConvexPolygon{3,3}
const Quadrilateral3D = ConvexPolygon{4,3}

Base.@propagate_inbounds function Base.getindex(poly::ConvexPolygon, i::Integer)
    getfield(poly, :points)[i]
end

function ConvexPolygon{N}(v::SVector{N, Point{Dim, T}}) where {N, Dim, T}
    return ConvexPolygon{N, Dim, T}(v)
end
ConvexPolygon{N}(x...) where {N} = ConvexPolygon(SVector(x))
ConvexPolygon(x...) = ConvexPolygon(SVector(x))

isconvex(tri::Triangle) = true
function isconvex(poly::ConvexPolygon{N, 2}) where {N}
    # If each of the
    for i ∈ 1:N-2
        
end
