# @code_warntype checked 2021/11/19

# A line segment in 2D space defined by its two endpoints.
# For ray tracing purposes, the line starts at points[1] and ends at points[2]
struct LineSegment_2D{F <: AbstractFloat} <: Edge_2D{F}
    points::SVector{2, Point_2D{F}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
# @code_warntype checked 2021/11/19
LineSegment_2D(p₁::Point_2D, p₂::Point_2D) = LineSegment_2D(SVector(p₁, p₂))

# Base
# -------------------------------------------------------------------------------------------------
Base.broadcastable(l::LineSegment_2D) = Ref(l)

# Methods
# -------------------------------------------------------------------------------------------------
# Interpolation
# l(0) yields points[1], and l(1) yields points[2]
# @code_warntype checked 2021/11/19
function (l::LineSegment_2D{F})(r::R) where {F <: AbstractFloat, R <: Real}
    return l.points[1] + F(r) * (l.points[2] - l.points[1])
end

# @code_warntype checked 2021/11/19
arc_length(l::LineSegment_2D) = distance(l.points[1], l.points[2])

# @code_warntype checked 2021/11/19
function intersect(l₁::LineSegment_2D{F}, l₂::LineSegment_2D{F}) where {F <: AbstractFloat}
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
    # Note that the same approach works in 3D and
    # "If the lines are skew, s and r represent the parameters of the points of closest
    # approach" - Intersection of two lines in three-space, Ronald Goldman, in Graphics
    # Gems by Andrew S. Glassner.
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
    # othersize just comparing abs(v⃗ × u⃗) to some fixed quantity causes problems
    ϵ = parametric_coordinate_ϵ
    θₚ = LineSegment_2D_parallel_θ
    v⃗ = l₁.points[2] - l₁.points[1]
    u⃗ = l₂.points[2] - l₂.points[1]
    if abs(v⃗ × u⃗) > θₚ * norm(v⃗) * norm(u⃗)
        w⃗ = l₂.points[1] - l₁.points[1]
        r = (w⃗ × u⃗)/(v⃗ × u⃗)
        p = l₁(r)
        s = ((r*v⃗ - w⃗) ⋅ u⃗)/(u⃗ ⋅ u⃗)
        return (-ϵ ≤ s ≤ 1 + ϵ) && (-ϵ ≤ r ≤ 1 + ϵ) ? (0x01, p) : (0x00, p)
    else
        return (0x00, Point_2D(F, 0))
    end
end

# Plot
# -------------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, l::LineSegment_2D)
        return convert_arguments(LS, [l.points[1].x, l.points[2].x])
    end
    
    function convert_arguments(LS::Type{<:LineSegments}, L::Vector{<:LineSegment_2D})
        return convert_arguments(LS, reduce(vcat, [[l.points[1].x, l.points[2].x] for l in L]))
    end
end
