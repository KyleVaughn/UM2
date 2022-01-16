# A quadratic segment that passes through three points: 𝘅₁, 𝘅₂, and 𝘅₃.
# The assumed relation of the points may be seen in the diagram below.
#                 ___𝘅₃___
#            ____/        \____
#        ___/                  \
#     __/                       𝘅₂
#   _/
#  /
# 𝘅₁
#
# NOTE: 𝘅₃ is not necessarily the midpoint, or even between 𝘅₁ and 𝘅₂, but the curve starts
# and ends and 𝘅₁ and 𝘅₂.
# 𝗾(r) = (2r-1)(r-1)𝘅₁ + r(2r-1)𝘅₂ + 4r(1-r)𝘅₃
# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
# Chapter 8, Advanced Data Representation, in the interpolation functions section
struct QuadraticSegment{N,T} <: Edge{N,T}
    points::SVector{3, Point{N,T}}
end

const QuadraticSegment_2D = QuadraticSegment{2}
const QuadraticSegment_3D = QuadraticSegment{3}

Base.@propagate_inbounds function Base.getindex(q::QuadraticSegment, i::Int)
    getfield(q, :points)[i]
end

# Constructors
# ---------------------------------------------------------------------------------------------
function QuadraticSegment(p₁::Point{N,T}, p₂::Point{N,T}, p₃::Point{N,T}) where {N,T}
    return QuadraticSegment{N,T}(SVector{3, Point{N,T}}(p₁, p₂, p₃))
end
function QuadraticSegment{N}(p₁::Point{N,T}, p₂::Point{N,T}, p₃::Point{N,T}) where {N,T}
    return QuadraticSegment{N,T}(SVector{3, Point{N,T}}(p₁, p₂, p₃))
end

# Methods
# ---------------------------------------------------------------------------------------------
# Interpolation
# q(0) = q[1], q(1) = q[2], q(1//2) = q[3]
function (q::QuadraticSegment)(r)
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    return Point((2r-1)*(r-1)q[1] + r*(2r-1)q[2] + 4r*(1-r)q[3])
end

arclength(q::QuadraticSegment) = arclength(q, Val(15))
function arclength(q::QuadraticSegment{N,T}, ::Val{NP}) where {N,T,NP}
    # Numerical integration is used.
    # (Gauss-Legengre quadrature)
    #     1                   NP
    # L = ∫ ‖(𝗗 ∘ 𝗾)(r)‖dr ≈ ∑ wᵢ‖(𝗗 ∘ 𝗾)(r)‖
    #     0                  i=1
    #
    w, r = gauss_legendre_quadrature(T, Val(NP))
    return sum(@. w * norm(𝗗(q, r)))
end

# Find the axis-aligned bounding box of the segment.
function boundingbox(q::QuadraticSegment_2D)
    # Find the r coordinates where ∂x/∂r = 0, ∂y/∂r = 0
    # We know ∇ q, so we can directly compute these values
    r_x = (3q[1][1] + q[2][1] - 4q[3][1])/(4(q[1][1] + q[2][1] - 2q[3][1]))
    if 0 < r_x < 1
        x_extreme = (2r_x-1)*(r_x-1)q[1][1] + r_x*(2r_x-1)q[2][1] + 4r_x*(1-r_x)q[3][1]
        xmin = min(q[1][1], q[2][1], x_extreme)
        xmax = max(q[1][1], q[2][1], x_extreme)
    else
        xmin = min(q[1][1], q[2][1])
        xmax = max(q[1][1], q[2][1])
    end

    r_y = (3q[1][2] + q[2][2] - 4q[3][2])/(4(q[1][2] + q[2][2] - 2q[3][2]))
    if 0 < r_y < 1
        y_extreme = (2r_y-1)*(r_y-1)q[1][2] + r_y*(2r_y-1)q[2][2] + 4r_y*(1-r_y)q[3][2]
        ymin = min(q[1][2], q[2][2], y_extreme)
        ymax = max(q[1][2], q[2][2], y_extreme)
    else
        ymin = min(q[1][2], q[2][2])
        ymax = max(q[1][2], q[2][2])
    end
    return AABB_2D(Point_2D(xmin, ymin), Point_2D(xmax, ymax))
end

# Return the derivative of q, evalutated at r
derivative(q::QuadraticSegment, r) = (4r - 3)*(q[1] - q[3]) + (4r - 1)*(q[2] - q[3])

# Return the Jacobian of q, evalutated at r
jacobian(q::QuadraticSegment, r) = derivative(q, r) 

# Return if the point is left of the quadratic segment
#   p    ^
#   ^   /
# v⃗ |  / u⃗
#   | /
#   o
function isleft(p::Point, q::QuadraticSegment)
    if isstraight(q) || p ∉  boundingbox(q)
        # We don't need to account for curvature if q is straight or p is outside
        # q's bounding box
        𝘂 = q[2] - q[1]
        𝘃 = p - q[1]
    else
        # Get the nearest point on q to p.
        # Construct vectors from a point on q (close to p_near) to p_near and p. 
        # Use the cross product of these vectors to determine if p isleft.
        r, p_near = nearest_point(p, q)
        
        if r < 1e-6 || 1 < r # If r is small or beyond the valid range, just use q[2]
            𝘂 = q[2] - q[1]
            𝘃 = p - q[1]
        else # otherwise use a point on q, close to p_near
            q_base = q(0.95r)
            𝘂 = p_near - q_base
            𝘃 = p - q_base
        end
    end
    return 𝘂 × 𝘃 > 0
end

# If the quadratic segment is effectively linear
@inline function isstraight(q::QuadraticSegment_2D)
    # u⃗ × v⃗ = |u⃗||v⃗|sinθ
    return abs((q[3] - q[1]) × (q[2] - q[1])) < 1e-8
end

function intersect(l::LineSegment, q::QuadraticSegment)
    ϵ = parametric_coordinate_ϵ
    if isstraight(q) # Use line segment intersection.
        # See LineSegment for the math behind this.
        v⃗ = l[2] - l[1]
        u⃗ = q[2] - q[1]
        vxu = v⃗ × u⃗ 
        # Parallel or collinear lines, return.
        1e-8 < abs(vxu) || return (0x00000000, SVector(Point(), Point()))
        w⃗ = q[1] - l[1]
        # Delay division until r,s are verified
        if 0 <= vxu 
            lowerbound = (-ϵ)vxu
            upperbound = (1 + ϵ)vxu
        else
            upperbound = (-ϵ)vxu
            lowerbound = (1 + ϵ)vxu
        end 
        r_numerator = w⃗ × u⃗ 
        s_numerator = w⃗ × v⃗ 
        if (lowerbound ≤ r_numerator ≤ upperbound) && (lowerbound ≤ s_numerator ≤ upperbound) 
            return (0x00000001, SVector(l(s_numerator/vxu), Point()))
        else
            return (0x00000000, SVector(Point(), Point()))
        end 
    else
        # q(r) = (2r-1)(r-1)𝘅₁ + r(2r-1)𝘅₂ + 4r(1-r)𝘅₃
        # q(r) = 2r²(𝘅₁ + 𝘅₂ - 2𝘅₃) + r(-3𝘅₁ - 𝘅₂ + 4𝘅₃) + 𝘅₁
        # Let D⃗ = 2(𝘅₁ + 𝘅₂ - 2𝘅₃), E⃗ = (-3𝘅₁ - 𝘅₂ + 4𝘅₃), F⃗ = x₁
        # q(r) = r²D⃗ + rE⃗ + F⃗
        # l(s) = 𝘅₄ + sw⃗
        # If D⃗ × w⃗ ≠ 0
        #   𝘅₄ + sw⃗ = r²D⃗ + rE⃗ + F⃗
        #   sw⃗ = r²D⃗ + rE⃗ + (F⃗ - 𝘅₄)
        #   0 = r²(D⃗ × w⃗) + r(E⃗ × w⃗) + (F⃗ - 𝘅₄) × w⃗
        #   Let A = (D⃗ × w⃗), B = (E⃗ × w⃗), C = (F⃗ - 𝘅₄) × w⃗
        #   0 = Ar² + Br + C
        #   r = (-B - √(B²-4AC))/2A, -B + √(B²-4AC))/2A)
        #   s = ((q(r) - p₄)⋅w⃗/(w⃗ ⋅ w⃗)
        #   r is invalid if:
        #     1) A = 0
        #     2) B² < 4AC
        #     3) r < 0 or 1 < r   (Curve intersects, segment doesn't)
        #   s is invalid if:
        #     1) s < 0 or 1 < s   (Line intersects, segment doesn't)
        # If D⃗ × w⃗ = 0, there is only one intersection and the equation reduces to line
        # intersection.
        npoints = 0x00000000
        p₁ = Point()
        p₂ = Point()
        D⃗ = 2(q[1] +  q[2] - 2q[3])
        E⃗ =  4q[3] - 3q[1] -  q[2]
        w⃗ = l[2] - l[1]
        A = D⃗ × w⃗
        B = E⃗ × w⃗
        C = (q[1] - l[1]) × w⃗
        w = w⃗ ⋅ w⃗
        if abs(A) < 1e-8 
            # Line intersection
            # Can B = 0 if A = 0 for non-trivial 𝘅?
            r = -C/B
            (-ϵ ≤ r ≤ 1 + ϵ) || return 0x00000000, SVector(p₁, p₂)
            p₁ = q(r)
            s = (p₁ - l[1]) ⋅ w⃗/w
            if (-ϵ ≤ s ≤ 1 + ϵ)
                npoints = 0x00000001
            end
        elseif B^2 ≥ 4A*C
            # Quadratic intersection
            # The compiler seem seems to catch the √(B^2 - 4A*C), for common subexpression 
            # elimination, so leaving for readability
            r₁ = (-B - √(B^2 - 4A*C))/2A
            r₂ = (-B + √(B^2 - 4A*C))/2A
            if (-ϵ ≤ r₁ ≤ 1 + ϵ)
                p = q(r₁)
                s₁ = (p - l[1]) ⋅ w⃗/w
                if (-ϵ ≤ s₁ ≤ 1 + ϵ)
                    p₁ = p
                    npoints += 0x00000001
                end
            end
            if (-ϵ ≤ r₂ ≤ 1 + ϵ)
                p = q(r₂)
                s₂ = (p - l[1]) ⋅ w⃗/w
                if (-ϵ ≤ s₂ ≤ 1 + ϵ)
                    p₂ = p
                    npoints += 0x00000001
                end
            end
            if npoints === 0x00000001 && p₁ === Point()
                p₁ = p₂
            end
        end
        return npoints, SVector(p₁, p₂)
    end
end

nearest_point(p::Point, q::QuadraticSegment) = nearest_point(p, q, 15)
# Return the closest point on the curve to point p, along with the value of r such that 
# q(r) = p_nearest
# Uses at most max_iters iterations of Newton-Raphson
function nearest_point(p::Point, q::QuadraticSegment{N,T}, max_iters::Int64) where {N,T}
    r = 1//2 + inv(𝗝(q, 1//2))*(p - q(1//2)) 
    for i ∈ 1:max_iters-1
        Δr = inv(𝗝(q, r))*(p - q(r)) 
        if abs(Δr) < 1e-7
            break
        end
        r += Δr
    end
    return r, q(r)
end

# Plot
# ---------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, q::QuadraticSegment_2D)
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
