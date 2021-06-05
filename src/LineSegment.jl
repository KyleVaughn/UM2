import Base: intersect
# A line segment in 3D space defined by its two endpoints.
# For ray tracing purposes, the line starts at points[1] and goes to points[2]
struct LineSegment{T <: AbstractFloat} <: Edge
    points::NTuple{2, Point{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
LineSegment(p₁::Point, p₂::Point) = LineSegment((p₁, p₂)) 

# Base
# -------------------------------------------------------------------------------------------------
Base.broadcastable(l::LineSegment) = Ref(l)

# Methods
# -------------------------------------------------------------------------------------------------
(l::LineSegment)(r::T) where {T <: AbstractFloat} = l.points[1] + r * (l.points[2] - l.points[1])

function intersect(l₁::LineSegment{T}, l₂::LineSegment{T}) where {T <: AbstractFloat}
    # NOTE: Doesn't work for colinear lines. (v⃗ × u⃗ = 0⃗)
    #
    # Using the equation of a line in parametric form
    # For l₁ = x⃗₁ + rv⃗ and l₂ = x⃗₂ + su⃗
    # x⃗₁ + rv⃗ = x⃗₂ + su⃗                             subtracting x⃗₁ from both sides
    # rv⃗ = (x⃗₂-x⃗₁) + su⃗                             w⃗ = x⃗₂-x⃗₁                                   
    # rv⃗ = w⃗ + su⃗                                   cross product with u⃗ (distributive)
    # r(v⃗ × u⃗) = w⃗ × u⃗ + s(u⃗ × u⃗)                   u⃗ × u⃗ = 0
    # r(v⃗ × u⃗) = w⃗ × u⃗                              dot product v⃗ × u⃗ to each side
    # r(v⃗ × u⃗) ⋅(v⃗ × u⃗) = (w⃗ × u⃗) ⋅(v⃗ × u⃗)          divide by (v⃗ × u⃗) ⋅(v⃗ × u⃗)
    # r = [(w⃗ × u⃗) ⋅(v⃗ × u⃗)]/[(v⃗ × u⃗) ⋅(v⃗ × u⃗)]
    # Note that if the lines are parallel or collinear, v⃗ × u⃗ = 0
    # We need to ensure r, s ∈ [0, 1]. Verifying this condition for r is simple, but we need to
    # solve for s as well.
    # x⃗₂ + su⃗ = x⃗₁ + rv⃗                              subtracting x⃗₂ from both sides
    # su⃗ = -w⃗ + rv⃗                                   we see that each element must satisfy
    # s(u⃗ ⋅ u⃗) = (-w⃗ + rv⃗) ⋅ u⃗                       hence
    # s = (rv⃗ - w⃗) ⋅ u⃗/(u⃗ ⋅ u⃗)
    # If the lines are skew, s and r represent the parameters of the points of closest 
    # approach - Intersection of two lines in three-space, Ronald Goldman, in Graphics
    # Gems by Andrew S. Glassner.
    v⃗ = l₁.points[2] - l₁.points[1]
    u⃗ = l₂.points[2] - l₂.points[1]
    w⃗ = l₂.points[1] - l₁.points[1]
    r = ((w⃗ × u⃗) ⋅ (v⃗ × u⃗))/((v⃗ × u⃗) ⋅ (v⃗ × u⃗))
    p = l₁(r)
    s = (r*v⃗ - w⃗) ⋅ u⃗/(u⃗ ⋅ u⃗)
    return (0 ≤ s ≤ 1) && (0 ≤ r ≤ 1) ? (true, p) : (false, p)
end

arc_length(l::LineSegment) = distance(l.points[1], l.points[2])
