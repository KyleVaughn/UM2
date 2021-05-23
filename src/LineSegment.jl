import Base: intersect
# A line segment in 3D space defined by its two endpoints.
struct LineSegment{T <: AbstractFloat} <: Edge
    p₁::Point{T}
    p₂::Point{T}
end

# Base methods
# -------------------------------------------------------------------------------------------------
Base.broadcastable(l::LineSegment) = Ref(l)

# Methods
# -------------------------------------------------------------------------------------------------
arc_length(l::LineSegment) = distance(l.p₁, l.p₂)
(l::LineSegment)(t::T) where {T <: AbstractFloat} = l.p₁ + t * (l.p₂ - l.p₁)
midpoint(l::LineSegment) = l(0.5)
function intersect(l₁::LineSegment, l₂::LineSegment)
    # NOTE: Doesn't work for colinear lines. (v⃗ × u⃗ = 0⃗)
    #
    # Using the equation of a line in parametric form
    # For l₁ = x⃗₁ + tv⃗ and l₂ = x⃗₂ + su⃗
    # x⃗₁ + tv⃗ = x⃗₂ + su⃗                             subtracting x⃗₁ from both sides
    # tv⃗ = (x⃗₂-x⃗₁) + su⃗                             w⃗ = x⃗₂-x⃗₁                                   
    # tv⃗ = w⃗ + su⃗                                   cross product with u⃗ (distributive)
    # t(v⃗ × u⃗) = w⃗ × u⃗ + s(u⃗ × u⃗)                   u⃗ × u⃗ = 0
    # t(v⃗ × u⃗) = w⃗ × u⃗                              dot product v⃗ × u⃗ to each side
    # t(v⃗ × u⃗) ⋅(v⃗ × u⃗) = (w⃗ × u⃗) ⋅(v⃗ × u⃗)          divide by (v⃗ × u⃗) ⋅(v⃗ × u⃗)
    # t = [(w⃗ × u⃗) ⋅(v⃗ × u⃗)]/[(v⃗ × u⃗) ⋅(v⃗ × u⃗)]
    # Note that if the lines are parallel or collinear, v⃗ × u⃗ = 0
    # We need to ensure t, s ∈ [0, 1]. Verifying this condition for t is simple, but we need to
    # solve for s as well.
    # x⃗₂ + su⃗ = x⃗₁ + tv⃗                              subtracting x⃗₂ from both sides
    # su⃗ = -w⃗ + tv⃗                                   we see that each element must satisfy
    # s(u⃗ ⋅ u⃗) = (-w⃗ + tv⃗) ⋅ u⃗                       hence
    # s = (tv⃗ - w⃗) ⋅ u⃗/(u⃗ ⋅ u⃗)
    # "If the lines are skew, s and t represent the parameters of the points of closest 
    # approach" - Intersection of two lines in three-space, Ronald Goldman, in Graphics
    # Gems by Andrew S. Glassner.
    v⃗ = l₁.p₂ - l₁.p₁
    u⃗ = l₂.p₂ - l₂.p₁
    w⃗ = l₂.p₁ - l₁.p₁
    t = ((w⃗ × u⃗) ⋅ (v⃗ × u⃗))/((v⃗ × u⃗) ⋅ (v⃗ × u⃗))
    p = l₁(t)
    s = (t*v⃗ - w⃗) ⋅ u⃗/(u⃗ ⋅ u⃗)
    return (0.0 ≤ s ≤ 1.0) && (0.0 ≤ t ≤ 1.0) ? (true, p) : (false, p)
end

function is_left(p₃::Point{T}, l::LineSegment{T}; 
        n̂::Point=Point(T(0), T(0), T(1))) where {T <: AbstractFloat} 
    # The line segment is defined by the line from p₁ to p₂.
    #     p₃
    #
    #
    #
    # p₁----------------p₂
    # If we define u⃗ = p₂-p₁ and v⃗ = p₃-p₁ we may test if a point is_left of the 
    # line using the cross product. See diagram below.
    #     ^ v⃗
    #    /
    #   /
    #  / θ
    # p₁----------------> u⃗    
    # u⃗ × v⃗ = |u⃗||v⃗| sin(θ)n⃗, where n⃗ is the unit vector perpendicular to the plane 
    # containing u⃗ and v⃗. However, the orientation of the plane is ambiguous, hence 
    # we need n̂ provided or assumed. For vectors in the x-y plane, we assume 
    # n̂ = (0, 0, 1). Since, |u⃗|, |v⃗| ≥ 0, then the point p₃ is left of the line if 
    # θ ∈ (0, π) -- equivalently if sin(θ) > 0. 
    # Hence our is_left condition is if the sign of the components of u⃗ × v⃗ and n̂ match
    u⃗ = l.p₂-l.p₁
    v⃗ = p₃-l.p₁
    return sign.((u⃗ × v⃗).coord) == sign.(n̂.coord)
end
