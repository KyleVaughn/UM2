# A line segment in 2D space defined by its two endpoints.
# For ray tracing purposes, the line starts at points[1] and ends at points[2]
struct LineSegment_2D <: Edge_2D
    points::SVector{2, Point_2D}
end

# Constructors
# -------------------------------------------------------------------------------------------------
LineSegment_2D(p₁::Point_2D, p₂::Point_2D) = LineSegment_2D(SVector(p₁, p₂))

# Base
# -------------------------------------------------------------------------------------------------
Base.broadcastable(l::LineSegment_2D) = Ref(l)
Base.getindex(l::LineSegment_2D, i::Int64) = l.points[i]
Base.firstindex(l::LineSegment_2D) = 1
Base.lastindex(l::LineSegment_2D) = 2

# Methods
# -------------------------------------------------------------------------------------------------
# Interpolation
# l(0) yields points[1], and l(1) yields points[2]
(l::LineSegment_2D)(r::Real) = l[1] + (l[2] - l[1])r
arclength(l::LineSegment_2D) = distance(l[1], l[2])

function intersect(l₁::LineSegment_2D, l₂::LineSegment_2D)
    # NOTE: Doesn't work for colinear/parallel lines. (v⃗ × u⃗ = 0). Also, the cross product
    # operator for 2D points returns a scalar (the 2-norm of the cross product).
    #
    # Using the equation of a line in parametric form
    # For l₁ = x⃗₁ + rv⃗ and l₂ = x⃗₂ + su⃗
    # x⃗₁ + rv⃗ = x⃗₂ + su⃗                             subtracting x⃗₁ from both sides
    # rv⃗ = (x⃗₂-x⃗₁) + su⃗                             w⃗ = x⃗₂-x⃗₁
    # rv⃗ = w⃗ + su⃗                                   cross product with u⃗ (distributive)
    # r(v⃗ × u⃗) = w⃗ × u⃗ + s(u⃗ × u⃗)                   u⃗ × u⃗ = 0
    # r(v⃗ × u⃗) = w⃗ × u⃗                              dot product v⃗ × u⃗ to each side
    # r = (w⃗ × u⃗)/(v⃗ × u⃗)
    # Note that if the lines are parallel or collinear, v⃗ × u⃗ = 0
    # We need to ensure r, s ∈ [0, 1]. Verifying this condition for r is simple, but we need to
    # solve for s as well.
    # x⃗₂ + su⃗ = x⃗₁ + rv⃗                              subtracting x⃗₂ from both sides
    # su⃗ = -w⃗ + rv⃗                                   we see that each element must satisfy
    # s(u⃗ ⋅ u⃗) = (-w⃗ + rv⃗) ⋅ u⃗                       hence
    # s = (rv⃗ - w⃗) ⋅ u⃗/(u⃗ ⋅ u⃗)
    #
    # To determine if the lines are parallel or collinear, accounting for floating point error,
    # we declare that all lines with angle less that θₚ between them are parallel or collinear.
    # Using v⃗ × u⃗ = |v⃗||u⃗|sin(θ), and knowing that for small θ, sin(θ) ≈ θ
    # We say all vectors such that
    #   abs(v⃗ × u⃗)
    #   --------- ≤ θₚ
    #     |v⃗||u⃗|
    # are parallel or collinear
    # We need to consider the magnitudes of the vectors due to the large range of segment sized used,
    # otherwise just comparing abs(v⃗ × u⃗) to some fixed quantity causes problems. Hence, we keep
    # |v⃗||u⃗|
    ϵ = parametric_coordinate_ϵ
    θₚ = LineSegment_2D_parallel_θ
    v⃗ = l₁[2] - l₁[1]
    u⃗ = l₂[2] - l₂[1]
    u = u⃗ ⋅ u⃗
    v = v⃗ ⋅ v⃗
    vxu = v⃗ × u⃗
    if vxu^2 > θₚ * v * u
        w⃗ = l₂[1] - l₁[1]
        r = w⃗ × u⃗/vxu
        (-ϵ ≤ r ≤ 1 + ϵ) || return (0x00000000, Point_2D())
        p = l₁(r)
        s = (r*v⃗ - w⃗) ⋅ u⃗/u
        return (-ϵ ≤ s ≤ 1 + ϵ) ? (0x00000001, p) : (0x00000000, p)
    else
        return (0x00000000, Point_2D())
    end
end

# Return if the point is left of the line segment
#   p    ^
#   ^   /
# v⃗ |  / u⃗
#   | /
#   o
function isleft(p::Point_2D, l::LineSegment_2D)
    u⃗ = l[2] - l[1]
    v⃗ = p - l[1]
    return u⃗ × v⃗ > 0
end

# Convert a vector of points to a vector of line segments, typically for visualization
function tolines(points::Vector{Point_2D})
    return [ LineSegment_2D(points[i], points[i+1]) for i = 1:length(points)-1 ]
end

# Plot
# -------------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, l::LineSegment_2D)
        return convert_arguments(LS, [l[1], l[2]])
    end

    function convert_arguments(LS::Type{<:LineSegments}, L::Vector{LineSegment_2D})
        return convert_arguments(LS, reduce(vcat, [[l[1], l[2]] for l in L]))
    end
end
