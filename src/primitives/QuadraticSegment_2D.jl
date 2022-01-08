# A quadratic segment in 2D space that passes through three points: x⃗₁, x⃗₂, and x⃗₃.
# The assumed relation of the points may be seen in the diagram below.
#                 ___x⃗₃___
#            ____/        \____
#        ___/                  \
#     __/                       x⃗₂
#   _/
#  /
# x⃗₁
#
# NOTE: x⃗₃ is not necessarily the midpoint, or even between x⃗₁ and x⃗₂.
# q(r) = (2r-1)(r-1)x⃗₁ + r(2r-1)x⃗₂ + 4r(1-r)x⃗₃
# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
# Chapter 8, Advanced Data Representation, in the interpolation functions section
struct QuadraticSegment_2D <: Edge_2D
    points::SVector{3, Point_2D}
end

# Constructors
# -------------------------------------------------------------------------------------------------
QuadraticSegment_2D(p₁::Point_2D, p₂::Point_2D, p₃::Point_2D) = QuadraticSegment_2D(SVector(p₁, p₂, p₃))

# Methods
# -------------------------------------------------------------------------------------------------
# Interpolation
# q(0) = q[1], q(1) = q[2], q(1//2) = q[3]
function (q::QuadraticSegment_2D)(r::Real)
    # See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 4th Edition
    # Chapter 8, Advanced Data Representation, in the interpolation functions section
    rₜ = Float64(r)
    return (2rₜ-1)*( rₜ-1)q[1] +
                rₜ*(2rₜ-1)q[2] +
               4rₜ*( 1-rₜ)q[3]
end

arclength(q::QuadraticSegment_2D) = arclength(q, Val(15))
function arclength(q::QuadraticSegment_2D, ::Val{N}) where {N}
    # Numerical integration is used.
    # (Gauss-Legengre quadrature)
    #     1                  N
    # L = ∫ ||∇ q⃗(r)||dr  ≈  ∑ wᵢ||∇ q⃗(rᵢ)||
    #     0                 i=1
    #
    w, r = gauss_legendre_quadrature(Val(N))
    return sum(@. w * norm(∇(q, r)))
end

# Find the axis-aligned bounding box of the segment.
function boundingbox(q::QuadraticSegment_2D)
    # Find the vertex to vertex vector that is the longest
    v⃗_12 = q[2] - q[1] # Vector from p₁ to p₂
    v⃗_13 = q[3] - q[1] # Vector from p₁ to p₃
    v⃗_23 = q[3] - q[2] # Vector from p₂ to p₃
    dsq_12 = v⃗_12 ⋅ v⃗_12 # Distance from p₁ to p₂ squared 
    dsq_13 = v⃗_13 ⋅ v⃗_13
    dsq_23 = v⃗_23 ⋅ v⃗_23
    x⃗₁ = Point_2D()
    x⃗₂ = Point_2D()
    u⃗ = Point_2D()
    v⃗ = Point_2D()
    # Majority of the time, dsq_12 is the largest, so this is tested first 
    if (dsq_13 ≤ dsq_12) && (dsq_23 ≤ dsq_12)
        u⃗ = v⃗_12
        v⃗ = v⃗_13
        u = dsq_12
        x⃗₁ = q[1]
        x⃗₂ = q[2]
    elseif (dsq_12 ≤ dsq_13) && (dsq_23 ≤ dsq_13)
        u⃗ = v⃗_13
        v⃗ = v⃗_12
        u = dsq_13
        x⃗₁ = q[1]
        x⃗₂ = q[3]
    else
        u⃗ = v⃗_23
        v⃗ = -v⃗_12
        u = dsq_23
        x⃗₁ = q[2]
        x⃗₂ = q[3]
    end
    # u⃗ is the longest vextex to vertex vector
    # v⃗ is the vector from u⃗[1] to the vertex not in u⃗
    # Example
    #                 ___p₃___
    #            ____/        \____
    #        ___/                  \
    #     __/                       p₂
    #   _/                   
    #  /
    # p₁
    # --------------------------------------- 
    # u⃗ = p₂ - p₁, v⃗ = p₃ - p₁
    #                    p₃
    #                 .
    #        v⃗     . 
    #           .                    p₂
    #        .              .   
    #     .       .    
    # p₁.               u⃗
    #------------------------------------------
    # We want to obtain the vector perpendicular to u⃗
    # from u⃗ to p₃. We call this h⃗. We obtain h⃗ using
    # a projection of v⃗ onto u⃗ (v⃗ᵤ). Then we see h⃗ = v⃗ - v⃗ᵤ
    #                 ___p₃___
    #            ____/    .   \____
    #        ___/          .       \
    #     __/               .       p₂
    #   _/                   p₁ + v⃗ᵤ
    #  /
    # p₁
    #
    v⃗ᵤ = (u⃗ ⋅v⃗/u) * u⃗ # Projection of v⃗ onto u⃗
    h⃗ = v⃗ - v⃗ᵤ # vector aligned bounding box height
    # We can now construct the bounding box using the quadrilateral (p₁, p₂, p₂ + h⃗, p₁ + h⃗)
    # This is the segment aligned bounding box. To get the axis-aligned bounding box,
    # we find the min and max x and y
    x⃗₃ = x⃗₁ + h⃗
    x⃗₄ = x⃗₂ + h⃗
    x = SVector(x⃗₁.x, x⃗₂.x, x⃗₃.x, x⃗₄.x )
    y = SVector(x⃗₁.y, x⃗₂.y, x⃗₃.y, x⃗₄.y )
    return Rectangle_2D(minimum(x), minimum(y), maximum(x), maximum(y))
end

nearest_point(p::Point_2D, q::QuadraticSegment_2D) = nearest_point(p, q, 30)
# Return the closest point on the curve to point p and the value of r such that q(r) = p_nearest
# Uses at most N iterations of Newton-Raphson
function nearest_point(p::Point_2D, q::QuadraticSegment_2D, N::Int64)
    r = 0.5
    Δr = 0.0
    for i = 1:N
        err = p - q(r)
        grad = ∇(q, r)
        if abs(grad[1]) > abs(grad[2])
            Δr = err[1]/grad[1]
        else
            Δr = err[2]/grad[2]
        end
        r += Δr
        if abs(Δr) < 1e-7
            break
        end
    end
    return r, q(r)
end

# Return the gradient of q, evalutated at r
function gradient(q::QuadraticSegment_2D, r::Real)
    rₜ = Float64(r)
    return (4rₜ - 3)*(q[1] - q[3]) +
           (4rₜ - 1)*(q[2] - q[3])
end

# Return the Laplacian of q, evalutated at r
function laplacian(q::QuadraticSegment_2D, r::Real)
    return 4(q[1] + q[2] - 2q[3])
end

function intersect(l::LineSegment_2D, q::QuadraticSegment_2D)
    ϵ = parametric_coordinate_ϵ
    if isstraight(q) # Use line segment intersection.
        # See LineSegment_2D for the math behind this.
        v⃗ = q[2] - q[1]
        u⃗ = l[2] - l[1]
        u = u⃗ ⋅ u⃗ 
        v = v⃗ ⋅ v⃗ 
        vxu = v⃗ × u⃗ 
        if vxu^2 > LineSegment_2D_parallel_θ² * v * u 
            w⃗ = l[1] - q[1]
            r = w⃗ × u⃗/vxu
            (-ϵ ≤ r ≤ 1 + ϵ) || return (0x00000000, SVector(Point_2D(), Point_2D()))
            p = q[1] + (q[2] - q[1])r
            s = (r*v⃗ - w⃗) ⋅ u⃗/u 
            if (-ϵ ≤ s ≤ 1 + ϵ)
                return (0x00000001, SVector(p, Point_2D()))
            else
                (0x00000000, SVector(p, Point_2D()))
            end
        else
            return (0x00000000, SVector(Point_2D(), Point_2D()))
        end 
    else
        # q(r) = (2r-1)(r-1)x⃗₁ + r(2r-1)x⃗₂ + 4r(1-r)x⃗₃
        # q(r) = 2r²(x⃗₁ + x⃗₂ - 2x⃗₃) + r(-3x⃗₁ - x⃗₂ + 4x⃗₃) + x⃗₁
        # Let D⃗ = 2(x⃗₁ + x⃗₂ - 2x⃗₃), E⃗ = (-3x⃗₁ - x⃗₂ + 4x⃗₃), F⃗ = x₁
        # q(r) = r²D⃗ + rE⃗ + F⃗
        # l(s) = x⃗₄ + sw⃗
        # If D⃗ × w⃗ ≠ 0
        #   x⃗₄ + sw⃗ = r²D⃗ + rE⃗ + F⃗
        #   sw⃗ = r²D⃗ + rE⃗ + (F⃗ - x⃗₄)
        #   0 = r²(D⃗ × w⃗) + r(E⃗ × w⃗) + (F⃗ - x⃗₄) × w⃗
        #   Let A = (D⃗ × w⃗), B = (E⃗ × w⃗), C = (F⃗ - x⃗₄) × w⃗
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
        p₁ = Point_2D()
        p₂ = Point_2D()
        D⃗ = 2(q[1] +  q[2] - 2q[3])
        E⃗ =  4q[3] - 3q[1] -  q[2]
        w⃗ = l[2] - l[1]
        A = D⃗ × w⃗
        B = E⃗ × w⃗
        C = (q[1] - l[1]) × w⃗
        w = w⃗ ⋅ w⃗
        if A^2 < w * QuadraticSegment_2D_1_intersection_ϵ^2
            # Line intersection
            # Can B = 0 if A = 0 for non-trivial x⃗?
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
            if npoints === 0x00000001 && p₁ === Point_2D()
                p₁ = p₂
            end
        end
        return npoints, SVector(p₁, p₂)
    end
end

# Return if the point is left of the quadratic segment
#   p    ^
#   ^   /
# v⃗ |  / u⃗
#   | /
#   o
function isleft(p::Point_2D, q::QuadraticSegment_2D)
    if isstraight(q) || p ∉  boundingbox(q)
        # We don't need to account for the curve if q is straight or p is outside
        # q's bounding box
        u⃗ = q[2] - q[1]
        v⃗ = p - q[1]
        return u⃗ × v⃗ > 0
    else
        # Get the nearest point on q to p.
        # Construct vectors from a point on q, close to p_near, to p_near and p. 
        # Use the cross product of these vectors to determine if p isleft.
        r, p_near = nearest_point(p, q)
        
        if r < 1e-6 || 1 < r # If r is small or beyond the valid range, just use q[2]
            u⃗ = q[2] - q[1]
            v⃗ = p - q[1]
        else # otherwise use a point on q, close to p_near
            q_base = q(0.95r)
            u⃗ = p_near - q_base
            v⃗ = p - q_base
        end
        return u⃗ × v⃗ > 0
    end
end

# If the quadratic segment is effectively linear
@inline function isstraight(q::QuadraticSegment_2D)
    # u⃗ × v⃗ = |u⃗||v⃗|sinθ
    return abs((q[3] - q[1]) × (q[2] - q[1])) < 1e-7
end

# # Get an upper bound on the derivative magnitude |∇ q⃗|
# function gradient_magnitude_upperbound(q::QuadraticSegment_2D)
#     # q'(r) = (4r-3)(q₁ - q₃) + (4r-1)(q₂ - q₃)
#     # |q'(r)| ≤ (4r-3)|(q₁ - q₃)| + (4r-1)|(q₂ - q₃)| ≤ 3Q₁ + Q₂
#     # Q₁ = max(|(q₁ - q₃)|, |(q₂ - q₃)|))
#     # Q₂ = min(|(q₁ - q₃)|, |(q₂ - q₃)|))
#     v_31 = norm(q[1] - q[3])
#     v_32 = norm(q[2] - q[3])
#     if v_31 > v_32
#         return 3v_31 +  v_32
#     else
#         return  v_31 + 3v_32
#     end
# end

# Plot
# -------------------------------------------------------------------------------------------------
if enable_visualization
    function convert_arguments(LS::Type{<:LineSegments}, q::QuadraticSegment_2D)
        rr = LinRange(0, 1, 15)
        points = q.(rr)
        coords = reduce(vcat, [[points[i], points[i+1]] for i = 1:length(points)-1])
        return convert_arguments(LS, coords)
    end

    function convert_arguments(LS::Type{<:LineSegments}, Q::Vector{QuadraticSegment_2D})
        point_sets = [convert_arguments(LS, q) for q in Q]
        return convert_arguments(LS, reduce(vcat, [pset[1] for pset in point_sets]))
    end
end
