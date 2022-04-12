"""
    Polyhedron(SVector{N, Point3D{T}})
    Polyhedron(p₁::Point3D{T}}, p₂::Point3D{T}}, ...)

Construct a `Polyhedron` with `N` vertices. Note this struct only supports the
polyhedra found in "The Visualization Toolkit: An Object-Oriented Approach to 3D 
Graphics, 4th Edition, Chapter 8, Advanced Data Representation". See this source for 
the indexing of points as well. Several aliases exist for convenience, e.g. 
Tetrahedron (`N`=4), Hexahedron (`N`=8), etc.
"""
struct Polyhedron{N, T} <:Cell{1, T}
    points::SVector{N, Point3D{T}}
end

# Aliases for convenience
const Tetrahedron = Polyhedron{4}
const Hexahedron  = Polyhedron{8}

Base.@propagate_inbounds function Base.getindex(poly::Polyhedron, i::Integer)
    getfield(poly, :points)[i]
end

function Polyhedron{N}(v::SVector{N, Point3D{T}}) where {N, T}
    return Polyhedron{N, T}(v)
end
Polyhedron{N}(x...) where {N} = Polyhedron(SVector(x))
Polyhedron(x...) = Polyhedron(SVector(x))
