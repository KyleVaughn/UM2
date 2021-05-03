# Constructors
# -------------------------------------------------------------------------------------------------
struct LineSegment
  p₁::Point
  p₂::Point
end

# Base methods
# -------------------------------------------------------------------------------------------------
Base.broadcastable(l::LineSegment) = Ref(l)

# Methods
# -------------------------------------------------------------------------------------------------
distance(l::LineSegment) = distance(l.p₁, l.p₂)
(l::LineSegment)(t) = l.p₁ + t * (l.p₂ - l.p₁)
midpoint(l::LineSegment) = l(0.5)
function intersects(l₁::LineSegment, l₂::LineSegment)
    # Using the equation of a line in parametric form
    # For l₁ = x₁ + tv⃗ and l₂ = x₂ + su⃗
    # x₁ + tv⃗ = x₂ + su⃗                                   subtracting x₁ from both sides
    # tv⃗ = (x₂-x₁) + su⃗                                   cross product with u⃗ (distributive)
    # t(v⃗ × u⃗) = (x₂-x₁) × u⃗ + s(u⃗ × u⃗)                   u⃗ × u⃗ = 0
    # t(v⃗ × u⃗) = (x₂-x₁) × u⃗                              dot product v⃗ × u⃗ to each side
    # t(v⃗ × u⃗) ⋅(v⃗ × u⃗) = [(x₂-x₁) × u⃗] ⋅(v⃗ × u⃗)          divide by (v⃗ × u⃗) ⋅(v⃗ × u⃗)
    # t = {[(x₂-x₁) × u⃗] ⋅(v⃗ × u⃗)}/{(v⃗ × u⃗) ⋅(v⃗ × u⃗)}     or
    # t = {[(x₂-x₁) × u⃗] ⋅(v⃗ × u⃗)}/||(v⃗ × u⃗) ⋅(v⃗ × u⃗)||²
    # Note that if the lines are parallel or collinear, v⃗ × u⃗ = 0
    # If the lines are skew (no intersection), s and t represent the parameters of the points of 
    # closest approach. See Andrew Glassner's Graphics Gems pg. 304 "Intersection of two lines in
    # three-space".
    # We need to ensure t, s ∈ [0, 1]. Verifying this condition for t is simple, but we need to 
    # solve for s as well.
    # x₂ + su⃗ = x₁ + tv⃗                                   subtracting x₂ from both sides
    # su⃗ = (x₁-x₂) + tv⃗                                   we see that each element must satisfy
    # suᵢ = (x₁ᵢ-x₂ᵢ) + tvᵢ                               hence
    # s = ((x₁ᵢ-x₂ᵢ) + tvᵢ)/uᵢ                            
    x₁ = l₁.p₁  
    v⃗ = l₁.p₂ - l₁.p₁
    x₂ = l₂.p₁  
    u⃗ = l₂.p₂ - l₂.p₁
    if (v⃗ × u⃗) ≈ zero(v⃗) # no intersection, exit and save time
        return false, zero(v⃗)
    else
        t = (((x₂-x₁) × u⃗) ⋅ (v⃗ × u⃗))/((v⃗ × u⃗) ⋅ (v⃗ × u⃗))
        pᵢ= l₁(t)
        if 0.0 ≤ t ≤ 1.0 # check t is valid
            # Find a non-zero component of u⃗ to compute s
            if !isapprox(u⃗.x, 0.0, atol=1.0e-6)
                i = 1
            elseif !isapprox(u⃗.y, 0.0, atol=1.0e-6) 
                i = 2    
            else
                i = 3
            end
            s = ((x₁[i]-x₂[i]) + t*v⃗[i])/u⃗[i]
            if 0.0 ≤ s ≤ 1.0 # check s is valid
                return true, pᵢ
            else
                return false, pᵢ           
            end
        else # if t is not valid, return p, the closest point to intersection on line 1.
          return false, pᵢ
        end
    end
end
function is_left(p₃::Point, l::LineSegment)
    # It is assumed that the point p₃ and the line l share the same z-coordinate 
    # (p₃.z == p₁.z == p₂.z)
    # The line segment is defined by the line from p₁ to p₂
    #     p₃
    #    
    #   
    #  
    # p₁----------------p₂    
    # If we define u⃗ = p₂-p₁ and v⃗ = p₃-p₁ we may test if a point is_left of the line using the cross
    # product. u⃗ × v⃗ = |u⃗||v⃗| sin(θ)n⃗, where n⃗ is the unit vector perpendicular to the plane
    # containing u⃗ and v⃗. Since n⃗ must be (0, 0, 1), due to u⃗ and v⃗ in the xy-plane, the point p₃ is 
    # left of the line if θ ∈ (0, π) -- equivalently if sin(θ) > 0. Since, |u⃗|, |v⃗| ≥ 0, then
    # sin(θ) > 0 ⟹   |u⃗||v⃗| sin(θ) > 0. Furthermore, sinze n⃗ᵢ= n⃗ⱼ= 0, we need only examine the
    # k̂ component of the u⃗ × v⃗ vector to determine the sign of |u⃗||v⃗| sin(θ). Hence our is_left
    # condition has now become "if k̂(u⃗ × v⃗) > 0, the point is left". Lastly, if we note
    # k̂(u⃗ × v⃗) = u⃗.x*v⃗.y - v⃗.x*u⃗.y, we may carry out the computation at last.
    #     ^ v⃗
    #    /
    #   /
    #  / θ
    # p₁----------------> u⃗ 
    l.p₁.z == l.p₂.z || throw(DomainError(p1.z, "Line points don't share the same z-coordinate")) 
    p₃.z == l.p₁.z || throw(DomainError(p₃.z, "Point doesn't share the same z-coordinate as line")) 
    u⃗ = l.p₂-l.p₁
    v⃗ = p₃-l.p₁ 
    if u⃗.x*v⃗.y - v⃗.x*u⃗.y > 0.0
        return true
    else
        return false 
    end
end
