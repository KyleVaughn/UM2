# A convex polygon defined by its vertices in counterclockwise order 
struct Polygon{N, Dim, T} <:Face{Dim, 1, T}
    points::SVector{N, Point{Dim, T}}
end

# Aliases for convenience
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

# Constructors
# ---------------------------------------------------------------------------------------------
function Polygon{N}(v::SVector{N, Point{Dim, T}}) where {N, Dim, T}
    return Polygon{N, Dim, T}(v)
end
Polygon{N}(x...) where {N} = Polygon(SVector(x))
Polygon(x...) = Polygon(SVector(x))

# Area
# ---------------------------------------------------------------------------------------------
# Uses the shoelace formula (https://en.wikipedia.org/wiki/Shoelace_formula)
function area(poly::Polygon{N, 2, T}) where {N, T}
    a = zero(T) # Scalar
    for i ∈ 1:N-1
        a += poly[i] × poly[i + 1]
    end
    a += poly[N] × poly[1]
    return norm(a)/2
end
# Not necessarily planar
#function area(poly::Polygon{N, 3, T}) where {N, T}
#    a = zero(SVector{3, T}) # Vector
#    for i ∈ 1:N-1
#        a += (poly[i] × poly[i + 1])
#    end
#    a += poly[N] × poly[1]
#    return norm(a)/2
#end

# Bounding box
# ---------------------------------------------------------------------------------------------
boundingbox(poly::Polygon) = boundingbox(poly.points)

# Centroid
# ---------------------------------------------------------------------------------------------
# (https://en.wikipedia.org/wiki/Centroid#Of_a_polygon)
function centroid(poly::Polygon{N, 2, T}) where {N, T}
    a = zero(T) # Scalar
    c = SVector{2,T}(0,0)
    for i ∈ 1:N-1
        subarea = poly[i] × poly[i+1]
        c += subarea*(poly[i] + poly[i+1])
        a += subarea
    end
    subarea = poly[N] × poly[1]
    c += subarea*(poly[N] + poly[1])
    a += subarea
    return Point(c/(3a))
end
# Not necessarily planar
## (https://en.wikipedia.org/wiki/Centroid#By_geometric_decomposition)
#function centroid(poly::Polygon{N, 3, T}) where {N, T}
#    # Decompose into triangles
#    a = zero(T)
#    c = SVector{3,T}(0,0,0)
#    for i ∈ 1:N-2
#        subarea = norm((poly[i+1] - poly[1]) × (poly[i+2] - poly[1]))
#        c += subarea*(poly[1] + poly[i+1] + poly[i+2])
#        a += subarea
#    end
#    return Point(c/(3a))
#end

# Point inside polygon
# ---------------------------------------------------------------------------------------------
# Test if a point is in a polygon for 2D points/polygons
function Base.in(p::Point2D, poly::Polygon{N, 2}) where {N}
    # Test if the point is to the left of each edge. 
    for i ∈ 1:N-1
        isleft(p, LineSegment2D(poly[i], poly[i + 1])) || return false
    end
    return isleft(p, LineSegment2D(poly[N], poly[1]))
end
# Not necessarily planar
#function Base.in(p::Point3D, poly::Polygon{N, 3}) where {N}
#    # Check if the point is even in the same plane as the polygon
#    plane = Hyperplane(poly[1], poly[2], poly[3])
#    p ∈ plane || return false
#    # Test that the point is to the left of each edge, oriented to the plane
#    for i = 1:N-1
#        isleft(p, LineSegment3D(poly[i], poly[i + 1]), plane) || return false
#    end
#    return isleft(p, LineSegment3D(poly[N], poly[1]), plane) 
#end

# Intersect
# ---------------------------------------------------------------------------------------------
#
# Intersection of a line segment and polygon in 2D
function intersect(l::LineSegment2D{T}, poly::Polygon{N, 2, T}
                       ) where {N,T <:Union{Float32, Float64}} 
    # Create the line segments that make up the polygon and intersect each one
    # until 2 unique points have been found
    p₁ = nan(Point2D{T}) 
    npoints = 0x0000
    for i ∈ 1:N-1
        hit, point = l ∩ LineSegment2D(poly[i], poly[i + 1]) 
        if hit
            if npoints === 0x0000 
                npoints = 0x0001
                p₁ = point
            elseif !(p₁ ≈ point)
                return true, SVector(p₁, point)
            end
        end
    end
    hit, point = l ∩ LineSegment2D(poly[N], poly[1]) 
    if hit
        if npoints === 0x0000 
            npoints = 0x0001
            p₁ = point
        elseif !(p₁ ≈ point)
            return true, SVector(p₁, point)
        end
    end
    return false, SVector(p₁, point)
end

# Triangulate
# ---------------------------------------------------------------------------------------------
# Return the vector of triangles corresponding to the polygon's triangulation
#
# Assumes polygon is convex and planar
function triangulate(poly::Polygon{N, 2, T}) where {N, T}
    triangles = MVector{N-2, Triangle{2, T}}(undef)
    if N === 3
        triangles[1] = poly
        return triangles
    end
    for i = 1:N-2
        triangles[i] = Triangle(poly[1], poly[i+1], poly[i+2])
    end
    return SVector(triangles.data)
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, poly::Polygon{N}) where {N}
        lines = [LineSegment(poly[(i-1) % N + 1],
                             poly[    i % N + 1]) for i = 1:N] 
        return convert_arguments(LS, lines)
    end

    function convert_arguments(LS::Type{<:LineSegments}, P::Vector{<:Polygon})
        point_sets = [convert_arguments(LS, poly) for poly ∈  P]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈ point_sets]))
    end

    function convert_arguments(M::Type{<:Mesh}, tri::Triangle)
        points = [tri[i].coord for i = 1:3]
        face = [1 2 3]
        return convert_arguments(M, points, face)
    end

    function convert_arguments(M::Type{<:Mesh}, T::Vector{<:Triangle})
        points = reduce(vcat, [[tri[i].coord for i = 1:3] for tri ∈  T])
        faces = zeros(Int64, length(T), 3)
        k = 1
        for i in 1:length(T), j = 1:3
            faces[i, j] = k
            k += 1
        end
        return convert_arguments(M, points, faces)
    end

    function convert_arguments(M::Type{<:Mesh}, poly::Polygon)
        return convert_arguments(M, triangulate(poly)) 
    end

    function convert_arguments(M::Type{<:Mesh}, P::Vector{<:Polygon})
        return convert_arguments(M, reduce(vcat, triangulate.(P)))            
    end
end
