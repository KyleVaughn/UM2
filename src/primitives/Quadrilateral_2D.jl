# Quadrilateral in 2D defined by its 4 vertices.
# NOTE: Quadrilaterals are assumed to be convex. This is because quadrilaterals must be convex
# to be valid finite elements, and we're generating meshes using a finite element mesh generator,
# so this seems reasonable. See link below.
# https://math.stackexchange.com/questions/2430691/jacobian-determinant-for-bi-linear-quadrilaterals
struct Quadrilateral_2D <: Face_2D
    # Counter clockwise order
    points::SVector{4, Point_2D}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Quadrilateral_2D(p₁::Point_2D, p₂::Point_2D, p₃::Point_2D, p₄::Point_2D
                ) = Quadrilateral_2D(SVector(p₁, p₂, p₃, p₄))

# Methods
# -------------------------------------------------------------------------------------------------
function (quad::Quadrilateral_2D)(r::Real, s::Real)
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    rₜ = Float64(r); sₜ = Float64(s)
    return (1 - rₜ)*(1 - sₜ)*quad[1] +
                 rₜ*(1 - sₜ)*quad[2] +
                       rₜ*sₜ*quad[3] +
                 (1 - rₜ)*sₜ*quad[4]
end

function area(quad::Quadrilateral_2D)
    # Using the convex quadrilateral assumption, just return the sum of the areas of the two
    # triangles that partition the quadrilateral. If the convex assumption ever changes, you
    # need to verify that the triangle pair partitions the quadrilateral. Choosing the wrong
    # pair overestimates the area, so just get the areas of both pairs of valid triangles and use
    return sum(area.(triangulate(quad)))
end

function centroid(quad::Quadrilateral_2D)
    tris = triangulate(quad)
    A₁ = area(tris[1])
    A₂ = area(tris[2])
    C₁ = centroid(tris[1])
    C₂ = centroid(tris[2])
    return (A₁*C₁ + A₂*C₂)/(A₁ + A₂)
end

function in(p::Point_2D, quad::Quadrilateral_2D)
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

function intersect(l::LineSegment_2D, quad::Quadrilateral_2D)
    # Create the 4 line segments that make up the quadrilateral and intersect each one
    edges = SVector(LineSegment_2D(quad[1], quad[2]),
                    LineSegment_2D(quad[2], quad[3]),
                    LineSegment_2D(quad[3], quad[4]),
                    LineSegment_2D(quad[4], quad[1]))
    ipoints = MVector(Point_2D(), Point_2D(), Point_2D())
    n_ipoints = 0x00000000
    # We need to account for 4 points returned due to vertex intersection
    # The only way we get 4 points though, is if 2 are redundant.
    # So, we return 3, which guarantees all unique points will be returned and the output
    # matched with the triangle output
    for k ∈  1:4
        npoints, point = l ∩ edges[k]
        if npoints === 0x00000001 && n_ipoints !== 0x00000003
            n_ipoints += 0x00000001 
            ipoints[n_ipoints] = point
        end
    end
    return n_ipoints, SVector(ipoints)
end

function triangulate(quad::Quadrilateral_2D)
    # Return the two triangles that partition the domain
    A, B, C, D = quad.points
    return SVector(Triangle_2D(A, B, C), Triangle_2D(C, D, A))
end

# Plot
# -------------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, quad::Quadrilateral_2D)
        l₁ = LineSegment_2D(quad[1], quad[2])
        l₂ = LineSegment_2D(quad[2], quad[3])
        l₃ = LineSegment_2D(quad[3], quad[4])
        l₄ = LineSegment_2D(quad[4], quad[1])
        lines = [l₁, l₂, l₃, l₄]
        return convert_arguments(LS, lines)
    end

    function convert_arguments(LS::Type{<:LineSegments}, Q::Vector{Quadrilateral_2D})
        point_sets = [convert_arguments(LS, quad) for quad ∈  Q]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset ∈  point_sets]))
    end

    function convert_arguments(M::Type{<:Mesh}, quad::Quadrilateral_2D)
        points = [quad[i] for i = 1:4]
        faces = [1 2 3;
                 3 4 1]
        return convert_arguments(M, points, faces)
    end

    function convert_arguments(M::Type{<:Mesh}, Q::Vector{Quadrilateral_2D})
        points = reduce(vcat, [[quad[i] for i = 1:4] for quad ∈  Q])
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
