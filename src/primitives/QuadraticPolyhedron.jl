"""
    QuadraticPolyhedron(SVector{N, Point3D{T}})
    QuadraticPolyhedron(p₁::Point3D{T}}, p₂::Point3D{T}}, ...)

Construct a `QuadraticPolyhedron` with `N` vertices. Note this struct only supports the
polyhedra found in "The Visualization Toolkit: An Object-Oriented Approach to 3D 
Graphics, 4th Edition, Chapter 8, Advanced Data Representation". See this source for 
the indexing of points as well. Several aliases exist for convenience, e.g. 
QuadraticTetrahedron (`N`=10), QuadraticHexahedron (`N`=20), etc.
"""
struct QuadraticPolyhedron{N, T} <:Cell{2, T}
    points::SVector{N, Point3D{T}}
end

# Aliases for convenience
const QuadraticTetrahedron = QuadraticPolyhedron{10}
const QuadraticHexahedron  = QuadraticPolyhedron{20}

Base.@propagate_inbounds function Base.getindex(poly::QuadraticPolyhedron, i::Integer)
    getfield(poly, :points)[i]
end

QuadraticPolyhedron{N}(v::SVector{N, Point3D{T}}) where {N, T} = 
    QuadraticPolyhedron{N, T}(v)
QuadraticPolyhedron{N}(x...) where {N} = QuadraticPolyhedron(SVector(x))
QuadraticPolyhedron(x...) = QuadraticPolyhedron(SVector(x))
