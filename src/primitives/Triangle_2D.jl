# Triangle in 2D defined by its 3 vertices.
struct Triangle_2D{F <: AbstractFloat} <: Face_2D{F}
    points::SVector{3, Point_2D{F}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Triangle_2D(p₁::Point_2D{F},
            p₂::Point_2D{F},
            p₃::Point_2D{F}) where {F <: AbstractFloat} = Triangle_2D(SVector(p₁, p₂, p₃))

# Methods (All type-stable)
# -------------------------------------------------------------------------------------------------
# Interpolation
function (tri::Triangle_2D{F})(r::R, s::S) where {F <: AbstractFloat,
                                                  R <: Real,
                                                  S <: Real}
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    return (1 - F(r) - F(s))*tri.points[1] + F(r)*tri.points[2] + F(s)*tri.points[3]
end

function area(tri::Triangle_2D)
    # A = bh/2
    # Let u⃗ = (v₂ - v₁), v⃗ = (v₃ - v₁)
    # b = |u⃗|
    # h = |sin(θ) v⃗|, where θ is the angle between u⃗ and v⃗
    # u⃗ × v⃗ = |u⃗||v⃗| sin(θ), hence
    # A = |u⃗ × v⃗|/2 = bh/2
    u⃗ = tri.points[2] - tri.points[1]
    v⃗ = tri.points[3] - tri.points[1]
    # 2D cross product returns a scalar
    return abs(u⃗ × v⃗)/2
end

function in(p::Point_2D{F}, tri::Triangle_2D{F}) where {F <: AbstractFloat}
    # If the point is to the left of every edge
    #  3<-----2
    #  |     ^
    #  | p  /
    #  |   /
    #  |  /
    #  v /
    #  1
    return is_left(p, LineSegment_2D(tri.points[1], tri.points[2])) &&
           is_left(p, LineSegment_2D(tri.points[2], tri.points[3])) &&
           is_left(p, LineSegment_2D(tri.points[3], tri.points[1]))
end

function intersect(l::LineSegment_2D{F}, tri::Triangle_2D{F}) where {F <: AbstractFloat}
    # Create the 3 line segments that make up the triangle and intersect each one
    line_segments = SVector(LineSegment_2D(tri.points[1], tri.points[2]),
                            LineSegment_2D(tri.points[2], tri.points[3]),
                            LineSegment_2D(tri.points[3], tri.points[1]))
    p₁ = Point_2D(F, 0)
    p₂ = Point_2D(F, 0)
    ipoints = 0x00000000
    # We need to account for 3 points returned due to vertex intersection
    for i = 1:3
        npoints, point = l ∩ line_segments[i]
        if npoints === 0x00000001
            if ipoints === 0x00000000
                p₁ = point
                ipoints = 0x00000001
            elseif ipoints === 0x00000001 && (point ≉  p₁)
                p₂ = point
                ipoints = 0x00000002
            end
        end
    end
    return ipoints, SVector(p₁, p₂)
end
intersect(tri::Triangle_2D, l::LineSegment_2D) = intersect(l, tri)

function Base.show(io::IO, tri::Triangle_2D{F}) where {F <: AbstractFloat}
    println(io, "Triangle_2D{$F}(")
    for i = 1:3
        p = tri.points[i]
        println(io, "  $p,")
    end
    println(io, " )")
end

# Plot
# -------------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, tri::Triangle_2D)
        l₁ = LineSegment_2D(tri.points[1], tri.points[2])
        l₂ = LineSegment_2D(tri.points[2], tri.points[3])
        l₃ = LineSegment_2D(tri.points[3], tri.points[1])
        lines = [l₁, l₂, l₃]
        return convert_arguments(LS, lines)
    end
    
    function convert_arguments(LS::Type{<:LineSegments}, T::Vector{Triangle_2D})
        point_sets = [convert_arguments(LS, tri) for tri in T]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset in point_sets]))
    end
    
    function convert_arguments(M::Type{<:Mesh}, tri::Triangle_2D)
        points = [tri.points[i] for i = 1:3]
        face = [1 2 3]
        return convert_arguments(M, points, face)
    end
    
    # Yes, the type needs to be this specific, otherwise it tries to dispatch on a Makie routine
    function convert_arguments(M::Type{Mesh{Tuple{Vector{Triangle_2D{F}}}}},
                               T::Vector{Triangle_2D{F}}) where {F <: AbstractFloat}
        points = reduce(vcat, [[tri.points[i] for i = 1:3] for tri in T])
        faces = zeros(Int64, length(T), 3)
        k = 1
        for i in 1:length(T), j = 1:3
            faces[i, j] = k
            k += 1
        end
        return convert_arguments(M, points, faces)
    end
end
