# A polygon defined by its vertices in counterclockwise order 
struct Polygon{N,Dim,T} <:Face{Dim,1,T}
    points::SVector{N, Point{Dim,T}}
end

# Aliases for convenience
const Triangle        = Polygon{3}
const Quadrilateral   = Polygon{4}
const Pentagon        = Polygon{5}
const Hexagon         = Polygon{6}
const Heptagon        = Polygon{7}
const Octagon         = Polygon{8}
const Nonagon         = Polygon{9}
const Decagon         = Polygon{10}
# When the time comes for 3D, use metaprogramming/eval to export 2D/3D consts
const Triangle2D      = Polygon{3,2}
const Quadrilateral2D = Polygon{4,2}

Base.@propagate_inbounds function Base.getindex(poly::Polygon, i::Integer)
    getfield(poly, :points)[i]
end

# Constructors
# ---------------------------------------------------------------------------------------------
function Polygon{N}(v::SVector{N, Point{Dim,T}}) where {N,Dim,T}
    return Polygon{N,Dim,T}(v)
end
Polygon{N}(x...) where {N} = Polygon(SVector(x))
Polygon(x...) = Polygon(SVector(x))

# Methods
# ---------------------------------------------------------------------------------------------
# Shoelace formula (https://en.wikipedia.org/wiki/Shoelace_formula)
function area(poly::Polygon{N,Dim,T}) where {N,Dim,T}
    # This can be done with mapreduce, but mapreduce is substantially slower
    if Dim === 2
        a = T(0) # Scalar
    else
        a = Base.zero(Point{Dim,T}) # Vector
    end
    for i ∈ 1:N
        a += poly[(i - 1) % N + 1] × poly[i % N + 1]
    end
    return norm(a)/2
end
# We can simplify the above for triangles
area(tri::Triangle) = norm(tri[2] - tri[1] × tri[3] - tri[1])/2

# Centroid for polygons in the 2D plane
function centroid(poly::Polygon{N,2,T}) where {N,T}
    c = SVector{2,T}(0,0)
    a = T(0)
    for i ∈ 1:N-1
        subarea = poly[i] × poly[i+1]
        c += subarea*(poly[i] + poly[i+1])
        a += subarea
    end
    return Point(c/(3a))
end
# Use a faster method for triangles
centroid(tri::Triangle) = tri(1//3, 1//3)

# Test if a point is in a polygon for 2D points/polygons
function Base.in(p::Point2D, poly::Polygon{N,2,T}) where {N,T}
    # Test if the point is to the left of each edge. 
    bool = true
    for i ∈ 1:N
        if !isleft(p, LineSegment2D(poly[(i - 1) % N + 1], poly[i % N + 1]))
            bool = false
            break
        end
    end
    return bool
end

function Base.intersect(l::LineSegment2D{T}, poly::Polygon{N,2,T}
                       ) where {N,T <:Union{Float32, Float64}} 
    # Create the line segments that make up the triangle and intersect each one
    points = zeros(MVector{N,Point2D{T}})
    npoints = 0x0000
    for i ∈ 1:N
        hit, point = l ∩ LineSegment2D(poly[(i - 1) % N + 1], poly[i % N + 1]) 
        if hit
            npoints += 0x0001
            @inbounds points[npoints] = point
        end
    end
    return npoints, SVector(points)
end

# Cannot mutate BigFloats in an MVector, so we use a regular Vector
function Base.intersect(l::LineSegment2D{BigFloat}, poly::Polygon{N,2,BigFloat}) where {N} 
    # Create the line segments that make up the triangle and intersect each one
    points = zeros(Point2D{BigFloat}, N)
    npoints = 0x0000
    for i ∈ 1:N
        hit, point = l ∩ LineSegment2D(poly[(i - 1) % N + 1], poly[i % N + 1]) 
        if hit
            npoints += 0x0001
            @inbounds points[npoints] = point
        end
    end
    return npoints, SVector{N,Point2D{BigFloat}}(points)
end

# Interpolation
# ---------------------------------------------------------------------------------------------
# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
# Chapter 8, Advanced Data Representation, in the interpolation functions section
function (tri::Triangle)(r, s)
    return Point((1 - r - s)*tri[1] + r*tri[2] + s*tri[3])
end

function (quad::Quadrilateral)(r, s)
    return Point((1 - r)*(1 - s)*quad[1] + r*(1 - s)*quad[2] + r*s*quad[3] + (1 - r)*s*quad[4])
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, poly::Polygon{N}) where {N}
        lines = [LineSegment2D(poly[(i-1) % N + 1],
                               poly[    i % N + 1]) for i = 1:N] 
        return convert_arguments(LS, lines)
    end

    function convert_arguments(LS::Type{<:LineSegments}, P::Vector{<:Polygon})
        point_sets = [convert_arguments(LS, poly) for poly ∈  P]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
    end

    # Need to implement triangulation before this can be done for a general polygon
    # function convert_arguments(M::Type{<:Mesh}, poly::)
    # end
    #function convert_arguments(M::Type{<:Mesh}, T::Vector{<:Triangle})
    #    points = reduce(vcat, [[tri[i].coord for i = 1:3] for tri ∈  T])
    #    faces = zeros(Int64, length(T), 3)
    #    k = 1
    #    for i in 1:length(T), j = 1:3
    #        faces[i, j] = k
    #        k += 1
    #    end
    #    return convert_arguments(M, points, faces)
    #end
end
