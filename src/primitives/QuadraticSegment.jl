# A quadratic segment that passes through three points: 𝘅₁, 𝘅₂, and 𝘅₃.
# The segment satisfies:
# 𝗾(r) = (2r-1)(r-1)𝘅₁ + r(2r-1)𝘅₂ + 4r(1-r)𝘅₃, r ∈ [0,1]
# Equivalently, 𝗾(r) = r²𝘂 + r𝘃 + 𝘅₁, r ∈ [0,1] where:
# 𝘂 = 2(𝘅₁ + 𝘅₂ - 2𝘅₃) and 𝘃 = -(3𝘅₁ + 𝘅₂ - 4𝘅₃)
# The relation of the points may be seen in the diagram below.
#                 ___𝘅₃___
#            ____/        \____
#        ___/                  \
#     __/                       𝘅₂
#   _/
#  /
# 𝘅₁
#
# NOTE: 𝘅₃ is not necessarily the midpoint in real space, or even between 𝘅₁ and 𝘅₂, 
# but the curve starts at 𝘅₁, passes through 𝘅₃ at q(1/2), and ends at 𝘅₂.
struct QuadraticSegment{Dim, T} <:Edge{Dim, 2, T}
    points::SVector{3, Point{Dim, T}}
end

const QuadraticSegment2D = QuadraticSegment{2}

Base.@propagate_inbounds function Base.getindex(q::QuadraticSegment, i::Integer)
    getfield(q, :points)[i]
end
# Easily fetch 𝘂, 𝘃, in 𝗾(r) = r²𝘂 + r𝘃 + 𝘅₁
# All branches but the correct one are pruned by the compiler, so this is fast
# when called inside a function.
function Base.getproperty(q::QuadraticSegment, sym::Symbol)
    if sym === :𝘂
        return 2(q[1] + q[2] - 2q[3])
    elseif sym === :𝘃
        return 4q[3] - 3q[1] - q[2]
    elseif sym === :𝘅₁
        return q[1] 
    elseif sym === :𝘅₂
        return q[2] 
    elseif sym === :𝘅₃
        return q[3] 
    else # fallback to getfield
        return getfield(q, sym)
    end
end

# Constructors
# ---------------------------------------------------------------------------------------------
function QuadraticSegment(p₁::Point{Dim, T}, 
                          p₂::Point{Dim, T}, 
                          p₃::Point{Dim, T}) where {Dim, T}
    return QuadraticSegment{Dim, T}(SVector{3, Point{Dim, T}}(p₁, p₂, p₃))
end
function QuadraticSegment{Dim}(p₁::Point{Dim, T}, 
                               p₂::Point{Dim, T}, 
                               p₃::Point{Dim, T}) where {Dim, T}
    return QuadraticSegment{Dim, T}(SVector{3, Point{Dim, T}}(p₁, p₂, p₃))
end

# Methods
# ---------------------------------------------------------------------------------------------
# Interpolation
#
# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
# Chapter 8, Advanced Data Representation, in the interpolation functions section
# Note: 𝗾(0) = 𝘅₁, 𝗾(1) = 𝘅₂, 𝗾(1/2) = 𝘅₃
function (q::QuadraticSegment)(r)
    return Point(((2r-1)*(r-1))q.𝘅₁ + (r*(2r-1))q.𝘅₂ + (4r*(1-r))q.𝘅₃)
end

# Return the arc length of the quadratic segment
#
# The arc length integral may be reduced to an integral over the square root of a 
# quadratic polynomial using ‖𝘅‖ = √(𝘅 ⋅ 𝘅), which has an analytic solution.
#     1             1
# L = ∫ ‖𝗾′(r)‖dr = ∫ √(ar² + br + c) dr 
#     0             0
function arclength(q::QuadraticSegment)
    if isstraight(q)
        return distance(q.𝘅₁, q.𝘅₂)
    else
        𝘂 = q.𝘂
        𝘃 = q.𝘃
        a = 4(𝘂 ⋅ 𝘂)
        b = 4(𝘂 ⋅ 𝘃)
        c = 𝘃 ⋅ 𝘃
        # Compiler seems to catch the reused sqrt quantities for common subexpression
        # elimination, or computation is as quick as storage in a variable, so we 
        # leave the sqrts for readability
        l = ((2a + b)√(a + b + c) - b√c)/4a -
            (b^2 - 4a*c)/((2√a)^3)*log((2√a√(a + b + c) + (2a + b))/(2√a√c + b)) 
        return l 
    end
end

# Find the axis-aligned bounding box of the segment
#
# Find the extrema for x and y by finding the r_x such that dx/dr = 0 
# and r_y such that dy/dr = 0
# 𝗾(r) = r²𝘂 + r𝘃 + 𝘅₁
# 𝗾′(r) = 2r𝘂 + 𝘃 ⟹  r_x, r_y = -𝘃 ./ 2𝘂
# Compare the extrema with the segment's endpoints to find the AABB
function boundingbox(q::QuadraticSegment2D)
    𝘂 = q.𝘂
    𝘃 = q.𝘃
    r_x, r_y = 𝘃 ./ -2𝘂
    if 0 < r_x < 1
        x_stationary = (r_x^2)𝘂[1] + r_x*𝘃[1] + q.𝘅₁[1]
        xmin = min(q.𝘅₁.x, q.𝘅₂.x, x_stationary)
        xmax = max(q.𝘅₁.x, q.𝘅₂.x, x_stationary)
    else
        xmin = min(q.𝘅₁.x, q.𝘅₂.x)
        xmax = max(q.𝘅₁.x, q.𝘅₂.x)
    end

    if 0 < r_y < 1
        y_stationary = (r_y^2)𝘂[2] + r_y*𝘃[2] + q.𝘅₁[2]
        ymin = min(q.𝘅₁.y, q.𝘅₂.y, y_stationary)
        ymax = max(q.𝘅₁.y, q.𝘅₂.y, y_stationary)
    else
        ymin = min(q.𝘅₁.y, q.𝘅₂.y)
        ymax = max(q.𝘅₁.y, q.𝘅₂.y)
    end
    return AABB2D(Point2D(xmin, ymin), Point2D(xmax, ymax))
end

# Return the derivative of q, evalutated at r
# 𝗾′(r) = 2r𝘂 + 𝘃, which is simplified to below.
derivative(q::QuadraticSegment, r) = (4r - 3)*(q.𝘅₁ - q.𝘅₃) + (4r - 1)*(q.𝘅₂ - q.𝘅₃)

# Return the Jacobian of q, evalutated at r
jacobian(q::QuadraticSegment, r) = derivative(q, r) 

# If the point is left of the quadratic segment in the 2D plane. 
#   𝗽    ^
#   ^   /
# 𝘃 |  / 𝘂
#   | /
#   o
# If the segment is straight, or if the point is not within the bounding box of
# the segment, we can perform the isleft check with the straight line from the 
# segment's start point to the segment's end point.
# If these conditions aren't met, the segment's curvature must be accounted for.
# We find the point on the curve q that is nearest point to the point of interest. 
# Call this point q_near. We then perform the isleft check with the tangent vector 
# of q at q_near and the vector from q_near to p, p - q_near.
function isleft(p::Point2D, q::QuadraticSegment2D)
    if isstraight(q) || p ∉  boundingbox(q)
        𝘂 = q.𝘅₂ - q.𝘅₁
        𝘃 = p - q.𝘅₁
    else
        r, q_near = nearest_point(p, q)
        𝘂 = 𝗗(q, r)
        𝘃 = p - q_near
    end
    return 𝘂 × 𝘃 > 0
end

# If the quadratic segment is effectively linear
#
# Check the sign of the cross product of the vectors (𝘅₃ - 𝘅₁) and (𝘅₂ - 𝘅₁)
# If the line is straight, 𝘅₃ - 𝘅₁ = c(𝘅₂ - 𝘅₁) where c ∈ (0, 1), hence
# (𝘅₃ - 𝘅₁) × (𝘅₂ - 𝘅₁) = 𝟬
@inline function isstraight(q::QuadraticSegment)
    return norm((q.𝘅₃ - q.𝘅₁) × (q.𝘅₂ - q.𝘅₁)) < 1e-8
end

# Intersection between a linesegment and quadratic segment
#
# The quadratic segment: 𝗾(r) = r²𝘂 + r𝘃 + 𝘅₁
# The line segment: 𝗹(s) = 𝘅₄ + s𝘄
# 𝘅₄ + s𝘄 = r²𝘂 + r𝘃 + 𝘅₁
# s𝘄 = r²𝘂 + r𝘃 + (𝘅₁ - 𝘅₄)
# 𝟬 = r²(𝘂 × 𝘄) + r(𝘃 × 𝘄) + (𝘅₁ - 𝘅₄) × 𝘄
# The cross product of two vectors in the plane is a vector of the form (0, 0, k).
# Let a = (𝘂 × 𝘄)ₖ, b = (𝘃 × 𝘄)ₖ, c = ([𝘅₁ - 𝘅₄] × 𝘄)ₖ
# 0 = ar² + br + c
# If a = 0 
#   r = -c/b
# else
#   r = (-b ± √(b²-4ac))/2a
# We must also solve for s
# 𝘅₄ + s𝘄 = 𝗾(r)
# s𝘄 = 𝗾(r) - 𝘅₄
# s = ([𝗾(r) - 𝘅₄] ⋅𝘄 )/(𝘄 ⋅ 𝘄)
#
# r is invalid if:
#   1) b² < 4ac
#   2) r ∉ [0, 1]   (Curve intersects, segment doesn't)
# s is invalid if:
#   1) s ∉ [0, 1]   (Line intersects, segment doesn't)
function Base.intersect(l::LineSegment2D{T}, q::QuadraticSegment2D{T}) where {T}
    ϵ = T(5e-6) # Tolerance on r,s ∈ [-ϵ, 1 + ϵ]
    npoints = 0x0000
    p₁ = zero(Point2D{T})
    p₂ = zero(Point2D{T})
    if isstraight(q) # Use line segment intersection.
        # See LineSegment for the math behind this.
        𝘄 = q.𝘅₁ - l.𝘅₁
        𝘃 = q.𝘅₂ - q.𝘅₁
        z = l.𝘂 × 𝘃
        r = (𝘄 × 𝘃)/z
        s = (𝘄 × l.𝘂)/z
        if T(1e-8) < abs(z) && -ϵ ≤ r ≤ 1 + ϵ && -ϵ ≤ s ≤ 1 + ϵ
            npoints += 0x0001
        end
        return npoints, SVector(l(r), p₂)
    else
        𝘂 = q.𝘂 
        𝘃 = q.𝘃 
        𝘄 = l.𝘂
        a = 𝘂 × 𝘄 
        b = 𝘃 × 𝘄
        c = (q.𝘅₁ - l.𝘅₁) × 𝘄
        w² = 𝘄 ⋅ 𝘄 
        if abs(a) < T(1e-8)
            r = -c/b
            -ϵ ≤ r ≤ 1 + ϵ || return 0x0000, SVector(p₁, p₂)
            p₁ = q(r)
            s = (p₁ - l.𝘅₁)⋅𝘄 
            if -ϵ*w² ≤ s ≤ (1 + ϵ)w²
                npoints = 0x0001
            end
        elseif b^2 ≥ 4a*c
            r₁ = (-b - √(b^2 - 4a*c))/2a
            r₂ = (-b + √(b^2 - 4a*c))/2a
            valid_p₁ = false
            if -ϵ ≤ r₁ ≤ 1 + ϵ
                p₁ = q(r₁)
                s₁ = (p₁ - l.𝘅₁)⋅𝘄
                if -ϵ*w² ≤ s₁ ≤ (1 + ϵ)w²
                    npoints += 0x0001
                    valid_p₁ = true
                end
            end
            if -ϵ ≤ r₂ ≤ 1 + ϵ
                p₂ = q(r₂)
                s₂ = (p₂ - l.𝘅₁)⋅𝘄
                if -ϵ*w² ≤ s₂ ≤ (1 + ϵ)w²
                    npoints += 0x0001
                end
            end
            if npoints === 0x0001 && !valid_p₁ 
                p₁ = p₂
            end
        end
        return npoints, SVector(p₁, p₂)
    end
end

# Find the point on 𝗾(r) closest to the point of interest 𝘆. 
#
# Note: r ∈ [0, 1] is not necessarily true for this function, since it finds the minimizer
# of the function 𝗾(r), ∀r ∈ ℝ 
# Find r which minimizes ‖𝘆 - 𝗾(r)‖, where 𝗾(r) = r²𝘂 + r𝘃 + 𝘅₁. 
# This r also minimizes ‖𝘆 - 𝗾(r)‖²
# It can be shown that this is equivalent to finding the minimum of the quartic function
# ‖𝘆 - 𝗾(r)‖² = f(r) = a₄r⁴ + a₃r³ + a₂r² + a₁r + a₀
# The minimum of f(r) occurs when f′(r) = ar³ + br² + cr + d = 0, where
# 𝘄 = 𝘆 - 𝘅₁, a = 4(𝘂 ⋅ 𝘂), b = 6(𝘂 ⋅ 𝘃), c = 2[(𝘃 ⋅ 𝘃) - 2(𝘂 ⋅𝘄)], d = -2(𝘃 ⋅ 𝘄)
# A cubic function is guaranteed to have at least 1 real root, which may be found using
# Cardano's formula (https://en.wikipedia.org/wiki/Cubic_equation#Cardano's_formula)
function nearest_point(pt::Point, qseg::QuadraticSegment)
    𝘂 = qseg.𝘂
    𝘃 = qseg.𝘃
    𝘄 = pt - qseg[1]
    # f′(r) = ar³ + br² + cr + d = 0
    a = 4(𝘂 ⋅ 𝘂)
    b = 6(𝘂 ⋅ 𝘃)
    c = 2((𝘃 ⋅ 𝘃) - 2(𝘂 ⋅𝘄))   
    d = -2(𝘃 ⋅ 𝘄)
    # t³ + pt + q = 0
    # t = r - b/3a
    p = (3a*c - b^2)/3a^2
    q = (2b^3 - 9a*c*b + 27a^2*d)/27a^3
    R = (p/3)^3 + (q/2)^3 
    r = ∛(-q/2 + √(R)) + ∛(-q/2 - √(R)) - b/3a
    return r, qseg(r)
end

# nearest_point(p::Point, q::QuadraticSegment) = nearest_point(p, q, 15)
# # Return the closest point on the curve q to point p, along with the value of r such that 
# # q(r) = p_nearest
# # Uses at most max_iters iterations of Newton-Raphson
# function nearest_point(p::Point, q::QuadraticSegment{Dim, T}, max_iters::Int64) where {Dim, T}
#     r = T(1//2) + inv(𝗝(q, 1//2))*(p - q(1//2)) 
#     for i ∈ 1:max_iters-1
#         Δr = inv(𝗝(q, r))*(p - q(r)) 
#         if abs(Δr) < T(1e-7)
#             break
#         end
#         r += Δr
#     end
#     return r, q(r)
# end

# Random quadratic segment in the Dim-dimensional unit hypercube
function Base.rand(::Type{QuadraticSegment{Dim, F}}) where {Dim, F} 
    points = rand(Point{Dim, F}, 3)
    return QuadraticSegment(points[1], points[2], points[3])
end

# N random quadratic segments in the Dim-dimensional unit hypercube
function Base.rand(::Type{QuadraticSegment{Dim, F}}, N::Int64) where {Dim, F}
    return [ rand(QuadraticSegment{Dim, F}) for i ∈ 1:N ]
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, q::QuadraticSegment)
        rr = LinRange(0, 1, 15)
        points = q.(rr)
        coords = reduce(vcat, [[points[i], points[i+1]] for i = 1:length(points)-1])
        return convert_arguments(LS, coords)
    end

    function convert_arguments(LS::Type{<:LineSegments}, Q::Vector{<:QuadraticSegment})
        point_sets = [convert_arguments(LS, q) for q in Q]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset in point_sets]))
    end
end
