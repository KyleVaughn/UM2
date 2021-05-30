import Base: intersect, in
# A line segment in 3D space defined by its two endpoints.
# For ray tracing purposes, the line starts at points[1] and goes to points[2]
struct LineSegment{T <: AbstractFloat} <: Edge
    points::NTuple{2, Point{T}}
end

# Constructors
# -------------------------------------------------------------------------------------------------
LineSegment(p₁::Point, p₂::Point) =LineSegment((p₁, p₂)) 

# Base methods
# -------------------------------------------------------------------------------------------------
Base.broadcastable(l::LineSegment) = Ref(l)

# Methods
# -------------------------------------------------------------------------------------------------
arc_length(l::LineSegment) = distance(l.points[1], l.points[2])
(l::LineSegment)(r::T) where {T <: AbstractFloat} = l.points[1] + r * (l.points[2] - l.points[1])
midpoint(l::LineSegment{T}) where {T <: AbstractFloat} = l(T(1//2))
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

function AABB(l::LineSegment{T}) where {T <: AbstractFloat}
    # Axis-aligned bounding box in xy-plane
    xmin = min(l.points[1][1], l.points[2][1])
    xmax = max(l.points[1][1], l.points[2][1])
    ymin = min(l.points[1][2], l.points[2][2])
    ymax = max(l.points[1][2], l.points[2][2])
    return (xmin, ymin, xmax, ymax)
end

function in(p::Point{T}, l::LineSegment{T}) where {T <: AbstractFloat}
    # if p⃗ is on the line then,
    # l(r) = x⃗₁ + ru⃗, where u⃗ = x⃗₂ - x⃗₁
    # p⃗ = x⃗₁ + ru⃗
    # p⃗ - x⃗₁ = ru⃗
    # r = (p⃗ - x⃗₁) ⋅ u⃗/(u⃗ ⋅ u⃗)
    # If r ∈ [0, 1], then the point is within the infinite width cylinder aligned with
    # l(r) and bounded by the points x⃗₁, x⃗₂. To see if the point is actually on the line,
    # we need to check l(r) ≈ p
    u⃗ = l.points[2] - l.points[1]
    r = (p - l.points[1]) ⋅ u⃗/(u⃗ ⋅ u⃗)
    return (0 ≤ r ≤ 1) && l(r) ≈  p
end

#function is_left(p₃::Point{T}, l::LineSegment{T}; 
#        n̂::Point{T}=Point(T(0), T(0), T(1))) where {T <: AbstractFloat} 
#    # The line segment is defined by the line from points[1] to points[2].
#    #     p₃
#    #
#    #
#    #
#    # p₁----------------p₂
#    # If we define u⃗ = p₂-p₁ and v⃗ = p₃-p₁ we may test if a point is_left of the 
#    # line using the cross product. See diagram below.
#    #     ^ v⃗
#    #    /
#    #   /
#    #  / θ
#    # -----------------> u⃗    
#    # u⃗ × v⃗ = |u⃗||v⃗| sin(θ)n⃗, where n⃗ is the unit vector perpendicular to the plane 
#    # containing u⃗ and v⃗. However, the orientation of the plane is ambiguous, hence 
#    # we need n̂ provided or assumed. For vectors in the x-y plane, we assume 
#    # n̂ = (0, 0, 1). Since, |u⃗|, |v⃗| ≥ 0, then the point p₃ is left of the line if 
#    # θ ∈ (0, π) -- equivalently if sin(θ) > 0. 
#    # Hence our is_left condition is if the sign of the components of u⃗ × v⃗ and n̂ match
#    u⃗ = l.points[2]-l.points[1]
#    v⃗ = p₃-l.points[1]
#    return sign.((u⃗ × v⃗).coord) == sign.(n̂.coord)
#end
