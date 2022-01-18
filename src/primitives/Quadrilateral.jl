# A quadrilateral defined by its 4 vertices.
# NOTE: Quadrilaterals are assumed to be convex. This is because quadrilaterals must be 
# convex to be valid finite elements, and we're generating meshes using a finite element mesh 
# generator, so this seems reasonable. See link below.
# https://math.stackexchange.com/questions/2430691/jacobian-determinant-for-bi-linear-quadrilaterals
struct Quadrilateral{N,T} <: Face{N,T}
    # Counter clockwise order
    points::SVector{4, Point{N,T}}
end

const Quadrilateral_2D = Quadrilateral{2}
const Quadrilateral_3D = Quadrilateral{3}

Base.@propagate_inbounds function Base.getindex(q::Quadrilateral, i::Integer)
    getfield(q, :points)[i]
end

# Constructors
# ---------------------------------------------------------------------------------------------
function Quadrilateral(p₁::Point{N,T}, p₂::Point{N,T}, 
                       p₃::Point{N,T}, p₄::Point{N,T}) where {N,T}
    return Quadrilateral{N,T}(SVector{4, Point{N,T}}(p₁, p₂, p₃, p₄))
end
function Quadrilateral{N}(p₁::Point{N,T}, p₂::Point{N,T}, 
                          p₃::Point{N,T}, p₄::Point{N,T}) where {N,T}
    return Quadrilateral{N,T}(SVector{4, Point{N,T}}(p₁, p₂, p₃, p₄))
end

# Methods
# ---------------------------------------------------------------------------------------------
function (quad::Quadrilateral)(r, s)
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    return Point((1 - r)*(1 - s)*quad[1] + r*(1 - s)*quad[2] + r*s*quad[3] + (1 - r)*s*quad[4])
end

function area(quad::Quadrilateral)
    # Using the convex quadrilateral assumption, just return the sum of the areas of the two
    # triangles that partition the quadrilateral.
    return sum(area.(triangulate(quad)))
end

function centroid(quad::Quadrilateral)
    tris = triangulate(quad)
    A₁ = area(tris[1])
    A₂ = area(tris[2])
    C₁ = centroid(tris[1])
    C₂ = centroid(tris[2])
    return (A₁*C₁ + A₂*C₂)/(A₁ + A₂)
end

function Base.in(p::Point_2D, quad::Quadrilateral_2D)
    # If the point is to the left of every edge
    #  4<-----3
    #  |      ^
    #  | p    |
    #  |      |
    #  |      |
    #  v----->2
    #  1
    return isleft(p, LineSegment_2D(quad[1], quad[2])) &&
           isleft(p, LineSegment_2D(quad[2], quad[3])) &&
           isleft(p, LineSegment_2D(quad[3], quad[4])) &&
           isleft(p, LineSegment_2D(quad[4], quad[1]))
end

function Base.intersect(l::LineSegment_2D{T}, quad::Quadrilateral_2D{T}) where {T} 
    # Create the 4 line segments that make up the quadrilateral and intersect each one
    # We need to account for 4 points returned due to vertex intersection
    # The only way we get 4 points though, is if 2 are redundant.
    # So, we return 3, which guarantees all unique points will be returned and the output
    # matched with the triangle output
    p₁ = Point_2D{T}(0,0)
    p₂ = Point_2D{T}(0,0)
    p₃ = Point_2D{T}(0,0)
    npoints = 0x0000
    for i ∈ 1:4
        hit, point = l ∩ LineSegment_2D(quad[(i - 1) % 4 + 1], 
                                        quad[      i % 4 + 1]) 
        if hit 
            npoints += 0x0001
            if npoints === 0x0001
                p₁ = point
            elseif npoints === 0x0002
                p₂ = point
            else
                p₃ = point
            end
        end
    end 
    return npoints, SVector(p₁, p₂, p₃) 
end

function triangulate(quad::Quadrilateral)
    # Return the two triangles that partition the domain
    A, B, C, D = quad.points
    return SVector(Triangle(A, B, C), Triangle(C, D, A))
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, quad::Quadrilateral)
        l₁ = LineSegment(quad[1], quad[2])
        l₂ = LineSegment(quad[2], quad[3])
        l₃ = LineSegment(quad[3], quad[4])
        l₄ = LineSegment(quad[4], quad[1])
        lines = [l₁, l₂, l₃, l₄]
        return convert_arguments(LS, lines)
    end

    function convert_arguments(LS::Type{<:LineSegments}, Q::Vector{<:Quadrilateral})
        point_sets = [convert_arguments(LS, quad) for quad ∈  Q]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈  point_sets]))
    end

    function convert_arguments(M::Type{<:Mesh}, quad::Quadrilateral)
        points = [quad[i].coord for i = 1:4]
        faces = [1 2 3;
                 3 4 1]
        return convert_arguments(M, points, faces)
    end

    function convert_arguments(M::Type{<:Mesh}, Q::Vector{<:Quadrilateral})
        points = reduce(vcat, [[quad[i].coord for i = 1:4] for quad ∈  Q])
        faces = zeros(Int64, 2*length(Q), 3)
        j = 0
        for i in 1:2:2*length(Q)
            faces[i    , :] = [1 2 3] .+ j
            faces[i + 1, :] = [3 4 1] .+ j
            j += 4
        end
        return convert_arguments(M, points, faces)
    end
end
