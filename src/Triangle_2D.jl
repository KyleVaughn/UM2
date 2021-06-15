# Triangle_2D defined by its 3 vertices.

struct Triangle_2D{T <: AbstractFloat}
    points::NTuple{3, Point_2D{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
Triangle_2D(p₁::Point_2D{T}, 
         p₂::Point_2D{T}, 
         p₃::Point_2D{T}) where {T <: AbstractFloat} = Triangle_2D((p₁, p₂, p₃))

# Methods
# -------------------------------------------------------------------------------------------------
function (tri::Triangle_2D{T})(r::R, s::S) where {T <: AbstractFloat, R,S <: Real}
    return (1 - T(r) - T(s))*tri.points[1] + T(r)*tri.points[2] + T(s)*tri.points[3]
end

function area(tri::Triangle_2D{T}) where {T <: AbstractFloat}
    # A = bh/2
    # Let u⃗ = |v₂ - v₁|, v⃗ = |v₃ - v₁|
    # b = |u⃗|
    # h = |sin(θ) v⃗|, where θ is the angle between u⃗ and v⃗
    # u⃗ × v⃗ = |u⃗||v⃗| sin(θ), hence
    # A = |u⃗ × v⃗|/2 = bh/2
    u⃗ = tri.points[2] - tri.points[1] 
    v⃗ = tri.points[3] - tri.points[1] 
    return (u⃗ × v⃗)/2
end

function in(p::Point_2D{T}, tri::Triangle_2D{T}) where {T <: AbstractFloat}
   # If the point is within the plane of the triangle, then the point is only within the triangle
   # if the areas of the triangles formed by the point and each pair of two vertices sum to the 
   # area of the triangle. Division by 2 is dropped, since it cancels
   A₁ = abs((tri.points[1] - p) × (tri.points[2] - p))
   A₂ = abs((tri.points[2] - p) × (tri.points[3] - p))
   A₃ = abs((tri.points[3] - p) × (tri.points[1] - p))
   A  = abs((tri.points[2] - tri.points[1]) × (tri.points[3] - tri.points[1]))
   return A₁ + A₂ + A₃ ≈ A
end

# Plot
# -------------------------------------------------------------------------------------------------
function convert_arguments(P::Type{<:LineSegments}, tri::Triangle_2D)
    l₁ = LineSegment_2D(tri.points[1], tri.points[2])
    l₂ = LineSegment_2D(tri.points[2], tri.points[3])
    l₃ = LineSegment_2D(tri.points[3], tri.points[1])
    lines = [l₁, l₂, l₃]
    return convert_arguments(P, lines)
end

function convert_arguments(P::Type{<:LineSegments}, AT::AbstractArray{<:Triangle_2D})
    point_sets = [convert_arguments(P, tri) for tri in AT]
    return convert_arguments(P, reduce(vcat, [pset[1] for pset in point_sets]))
end

function convert_arguments(P::Type{<:Mesh}, tri::Triangle_2D)
    points = [tri.points[i].x for i = 1:3]
    face = [1 2 3]
    return convert_arguments(P, points, face)
end

function convert_arguments(MT::Type{Mesh{Tuple{Vector{Triangle_2D{T}}}}}, 
        AT::Vector{Triangle_2D{T}}) where {T <: AbstractFloat} 
    points = reduce(vcat, [[tri.points[i].x for i = 1:3] for tri in AT])
    faces = zeros(Int64, length(AT), 3) 
    k = 1
    for i in 1:length(AT), j = 1:3
        faces[i, j] = k
        k += 1
    end
    return convert_arguments(MT, points, faces)
end
