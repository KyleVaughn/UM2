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
    #     1             NP
    # L = ∫ ‖𝗾′(r)‖dr ≈ ∑ wᵢ‖𝗾′(r)‖
    #     0            i=1
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

function Base.intersect(l::LineSegment_2D{T}, q::QuadraticSegment_2D{T}) where {T}
    ϵ = T(5e-6)
    npoints = 0x0000
    p₁ = Point_2D{T}(0,0)
    p₂ = Point_2D{T}(0,0)
    if isstraight(q) # Use line segment intersection.
        # See LineSegment for the math behind this.
        𝘄 = q[1] - l.𝘅₁
        𝘃 = q[2] - q[1]
        z = l.𝘂 × 𝘃
        r = (𝘄 × 𝘃)/z
        s = (𝘄 × l.𝘂)/z
        if T(1e-8) < abs(z) && ϵ ≤ r && r ≤ 1 + ϵ && ϵ ≤ s && s ≤ 1 + ϵ
            npoints += 0x0001
        end
        return npoints, SVector(l(r), p₂)
    else
        # q(r) = (2r-1)(r-1)𝘅₁ + r(2r-1)𝘅₂ + 4r(1-r)𝘅₃
        # q(r) = 2r²(𝘅₁ + 𝘅₂ - 2𝘅₃) + r(-3𝘅₁ - 𝘅₂ + 4𝘅₃) + 𝘅₁
        # Let 𝘂 = 2(𝘅₁ + 𝘅₂ - 2𝘅₃), 𝘃 = (-3𝘅₁ - 𝘅₂ + 4𝘅₃)
        # q(r) = r²𝘂 + r𝘃 + 𝘅₁
        # l(s) = 𝘅₄ + s𝘄
        # If 𝘂 × 𝘄 ≠ 𝟬
        #   𝘅₄ + s𝘄 = r²𝘂 + r𝘃 + 𝘅₁
        #   s𝘄 = r²𝘂 + r𝘃 + (𝘅₁ - 𝘅₄)
        #   0 = r²(𝘂 × 𝘄) + r(𝘃 × 𝘄) + (𝘅₁ - 𝘅₄) × 𝘄
        #   # In 2D the cross product yields a scalar
        #   Let a = (𝘂 × 𝘄), b = (𝘃 × 𝘄), c = (𝘅₁ - 𝘅₄) × 𝘄
        #   0 = ar² + br + c
        #   r = (-b ± √(b²-4ac))/2a
        #   # We must also solve for s
        #   r²𝘂 + r𝘃 + 𝘅₁ = 𝘅₄ + s𝘄 
        #   s𝘄 = r²𝘂 + r𝘃 + (𝘅₁ - 𝘅₄)
        #   s(𝘄 × 𝘂) = r²(𝘂 × 𝘂) + r(𝘃 × 𝘂) + (𝘅₁ - 𝘅₄) × 𝘂
        #   -as = r(𝘃 × 𝘂) + c
        #   s = ((𝘂 × 𝘃)r - c)/a
        #   or
        #   s = ((q(r) - 𝘅₄)⋅𝘄/(𝘄 ⋅ 𝘄)
        #   r is invalid if:
        #     1) a = 0
        #     2) b² < 4ac
        #     3) r < 0 or 1 < r   (Curve intersects, segment doesn't)
        #   s is invalid if:
        #     1) s < 0 or 1 < s   (Line intersects, segment doesn't)
        # If a = 0, there is only one intersection and the equation reduces to line
        # intersection.
        𝘂 = 2(q[1] +  q[2] - 2q[3])
        𝘃 =  4q[3] - 3q[1] -  q[2]
        𝘄 = l.𝘂
        a = 𝘂 × 𝘄 
        b = 𝘃 × 𝘄
        c = (q[1] - l.𝘅₁) × 𝘄
        d = 𝘂 × 𝘃
        if abs(a) < 1e-8 
            # Line intersection
            r = -c/b
            (-ϵ ≤ r ≤ 1 + ϵ) || return 0x0000, SVector(p₁, p₂)
            s = (q(r) - l.𝘅₁)⋅𝘄 /(𝘄 ⋅ 𝘄)
            p₁ = l(s)
            if (-ϵ ≤ s ≤ 1 + ϵ)
                npoints = 0x0001
            end
        elseif b^2 ≥ 4a*c
            # Quadratic intersection
            disc = √(b^2 - 4a*c)
            r₁ = (-b - disc)/2a
            r₂ = (-b + disc)/2a
            if (-ϵ ≤ r₁ ≤ 1 + ϵ)
                p₁ = q(r₁)
                s₁ = (d*r₁ - c)/a
                if (-ϵ ≤ s₁ ≤ 1 + ϵ)
                    npoints += 0x0001
                end
            end
            if (-ϵ ≤ r₂ ≤ 1 + ϵ)
                p₂ = q(r₂)
                s₂ = (d*r₂ - c)/a
                if (-ϵ ≤ s₂ ≤ 1 + ϵ)
                    npoints += 0x0001
                end
            end
            if npoints === 0x0001 && p₁ === Point_2D{T}(0,0)
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
