"""
    ConvexPolyhedron(SVector{N, Point3D{T}})
    ConvexPolyhedron(p₁::Point3D{T}}, p₂::Point3D{T}}, ...)

Construct a `ConvexPolyhedron` with `N` vertices. Note this struct only supports the
polyhedra found in "The Visualization Toolkit: An Object-Oriented Approach to 3D 
Graphics, 4th Edition, Chapter 8, Advanced Data Representation". See this source for 
the indexing of points as well. Several aliases exist for convenience, e.g. 
Tetrahedron (`N`=4), Hexahedron (`N`=8), etc.
"""
struct ConvexPolyhedron{N, T} <:Cell{1, T}
    points::SVector{N, Point3D{T}}
end

# Aliases for convenience
const Tetrahedron = ConvexPolyhedron{4}
const Hexahedron  = ConvexPolyhedron{8}

Base.@propagate_inbounds function Base.getindex(poly::ConvexPolyhedron, i::Integer)
    getfield(poly, :points)[i]
end

function ConvexPolyhedron{N}(v::SVector{N, Point3D{T}}) where {N, T}
    return ConvexPolyhedron{N, T}(v)
end
ConvexPolyhedron{N}(x...) where {N} = ConvexPolyhedron(SVector(x))
ConvexPolyhedron(x...) = ConvexPolyhedron(SVector(x))
