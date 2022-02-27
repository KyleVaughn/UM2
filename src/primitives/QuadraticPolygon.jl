"""
    QuadraticPolygon(SVector{N, Point{Dim, T}})
    QuadraticPolygon(p₁::Point{Dim, T}}, p₂::Point{Dim, T}}, ...)

Construct a `QuadraticPolygon` with `N` counter-clockwise oriented vertices in 
`Dim`-dimensional space. Several aliases exist for convenience, e.g. 
QuadraticTriangle (`N`=6), QuadraticQuadrilateral (`N`=8), etc.

The ordering for vertices for a quadratic triangle is as follows:
p₁ = vertex A     
p₂ = vertex B     
p₃ = vertex C     
p₄ = point on the quadratic segment from A to B
p₅ = point on the quadratic segment from B to C
p₆ = point on the quadratic segment from C to A
"""
struct QuadraticPolygon{N, Dim, T} <:Face{Dim, 2, T}
    points::SVector{N, Point{Dim, T}}
end

# Aliases for convenience
const QuadraticTriangle        = QuadraticPolygon{6}
const QuadraticQuadrilateral   = QuadraticPolygon{8}
const QuadraticTriangle2D      = QuadraticPolygon{6,2}
const QuadraticQuadrilateral2D = QuadraticPolygon{8,2}
const QuadraticTriangle3D      = QuadraticPolygon{6,3}
const QuadraticQuadrilateral3D = QuadraticPolygon{8,3}

Base.@propagate_inbounds function Base.getindex(poly::QuadraticPolygon, i::Integer)
    getfield(poly, :points)[i]
end

QuadraticPolygon{N}(v::SVector{N, Point{Dim, T}}) where {N, Dim, T} = 
    QuadraticPolygon{N, Dim, T}(v)
QuadraticPolygon{N}(x...) where {N} = QuadraticPolygon(SVector(x))
QuadraticPolygon(x...) = QuadraticPolygon(SVector(x))
