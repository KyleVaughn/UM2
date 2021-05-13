# A line segment in 3D space defined by its two endpoints.
struct LineSegment{T <: AbstractFloat}
    p⃗₁::Point{T}
    p⃗₂::Point{T}
end

# Base methods
# -------------------------------------------------------------------------------------------------
Base.broadcastable(l::LineSegment) = Ref(l)

# Methods
# -------------------------------------------------------------------------------------------------
arc_length(l::LineSegment) = distance(l.p⃗₁, l.p⃗₂)
(l::LineSegment)(t) = l.p⃗₁ + t * (l.p⃗₂ - l.p⃗₁)
midpoint(l::LineSegment) = l(0.5)
function intersects(l₁::LineSegment, l₂::LineSegment)
    # Using the equation of a line in parametric form
    # For l₁ = x⃗₁ + tv⃗ and l₂ = x⃗₂ + su⃗
    # x⃗₁ + tv⃗ = x⃗₂ + su⃗                                   subtracting x⃗₁ from both sides
    # tv⃗ = (x⃗₂-x⃗₁) + su⃗                                   cross product with u⃗ (distributive)
    # t(v⃗ × u⃗) = (x⃗₂-x⃗₁) × u⃗ + s(u⃗ × u⃗)                   u⃗ × u⃗ = 0
    # t(v⃗ × u⃗) = (x⃗₂-x⃗₁) × u⃗                              dot product v⃗ × u⃗ to each side
    # t(v⃗ × u⃗) ⋅(v⃗ × u⃗) = [(x⃗₂-x⃗₁) × u⃗] ⋅(v⃗ × u⃗)          divide by (v⃗ × u⃗) ⋅(v⃗ × u⃗)
    # t = {[(x⃗₂-x⃗₁) × u⃗] ⋅(v⃗ × u⃗)}/{(v⃗ × u⃗) ⋅(v⃗ × u⃗)}     or
    # t = {[(x⃗₂-x⃗₁) × u⃗] ⋅(v⃗ × u⃗)}/||(v⃗ × u⃗) ⋅(v⃗ × u⃗)||²
    # Note that if the lines are parallel or collinear, v⃗ × u⃗ = 0
    # If the lines are skew (no intersection), s and t represent the parameters of the points of
    # closest approach. See Andrew Glassner's Graphics Gems pg. 304 "Intersection of two lines in
    # three-space".
    # We need to ensure t, s ∈ [0, 1]. Verifying this condition for t is simple, but we need to
    # solve for s as well.
    # x⃗₂ + su⃗ = x⃗₁ + tv⃗                                   subtracting x⃗₂ from both sides
    # su⃗ = (x⃗₁-x⃗₂) + tv⃗                                   we see that each element must satisfy
    # s(u⃗ ⋅ u⃗) = [(x⃗₁-x⃗₂) + tv⃗] ⋅ u⃗                       hence
    # s = [(x⃗₁-x⃗₂) + tv⃗] ⋅ u⃗/(u⃗ ⋅ u⃗)
    x⃗₁ = l₁.p⃗₁
    v⃗ = l₁.p⃗₂ - l₁.p⃗₁
    x⃗₂ = l₂.p⃗₁
    u⃗ = l₂.p⃗₂ - l₂.p⃗₁
    if (v⃗ × u⃗) ≈ zero(v⃗) # no intersection, exit and save time
        return false, Point(Inf, Inf, Inf)
    end
    t = (((x⃗₂-x⃗₁) × u⃗) ⋅ (v⃗ × u⃗))/((v⃗ × u⃗) ⋅ (v⃗ × u⃗))
    p⃗ᵢ= l₁(t)
    s = (((x⃗₁-x⃗₂) + t*v⃗) ⋅ u⃗ )/(u⃗ ⋅ u⃗)
    if (0.0 ≤ s ≤ 1.0) && (0.0 ≤ t ≤ 1.0)
        return true, p⃗ᵢ
    else
        # if t/s is not valid, still return p⃗ᵢ, the closest point to intersection on line 1.
        return false, p⃗ᵢ
    end
end

function is_left(p⃗₃::Point, l::LineSegment)
    # It is assumed that the point p⃗₃ and the line l share the same z-coordinate.
    # (p⃗₃[3] == p⃗₁[3] == p⃗₂[3])
    # The line segment is defined by the line from p⃗₁ to p⃗₂.
    #     p⃗₃
    #
    #
    #
    # p⃗₁----------------p⃗₂
    # If we define u⃗ = p⃗₂-p⃗₁ and v⃗ = p⃗₃-p⃗₁ we may test if a point is_left of the line using the cross
    # product. See diagram below.
    # u⃗ × v⃗ = |u⃗||v⃗| sin(θ)n⃗, where n⃗ is the unit vector perpendicular to the plane
    # containing u⃗ and v⃗. Since n⃗ must be (0, 0, 1), due to u⃗ and v⃗ in the xy-plane, the point p⃗₃ is
    # left of the line if θ ∈ (0, π) -- equivalently if sin(θ) > 0. Since, |u⃗|, |v⃗| ≥ 0, then
    # sin(θ) > 0 ⟹   |u⃗||v⃗| sin(θ) > 0. Furthermore, since n⃗ᵢ= n⃗ⱼ= 0, we need only examine the
    # k̂ component of the u⃗ × v⃗ vector to determine the sign of |u⃗||v⃗| sin(θ). Hence our is_left
    # condition has now become "if k̂(u⃗ × v⃗) > 0, the point is left". Lastly, if we note
    # k̂(u⃗ × v⃗) = u⃗[1]*v⃗[2] - v⃗[1]*u⃗[2], we may carry out the computation at last.
    #     ^ v⃗
    #    /
    #   /
    #  / θ
    # p⃗₁----------------> u⃗
    @debug l.p⃗₁[3] == l.p⃗₂[3] == p⃗₃[3] || throw(DomainError(p⃗₁[3],
                                                "Points don't share the same z-coordinate"))
    u⃗ = l.p⃗₂-l.p⃗₁
    v⃗ = p⃗₃-l.p⃗₁
    if u⃗[1]*v⃗[2] - v⃗[1]*u⃗[2] > 0.0
        return true
    else
        return false
    end
end
