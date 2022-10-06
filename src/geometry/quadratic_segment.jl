export QuadraticSegment,
       QuadraticSegment2,
       QuadraticSegment2f,
       QuadraticSegment2d

export interpolate_quadratic_segment,
       jacobian,
       quadratic_segment_jacobian,
       arclength,
       area_enclosed_by,
       area_enclosed_by_quadratic_segment,
       centroid_of_area_enclosed_by,
       centroid_of_area_enclosed_by_quadratic_segment,
       bounding_box,
       isleft

# QUADRATIC SEGMENT
# -----------------------------------------------------------------------------
#
# A quadratic segment represented by 3 vertices.
# These vertices are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#
# It is helpful to know:
#  Q(r) = r²𝗮 + 𝗯r + C,
# where
#  𝗮 = 2(P₁ + P₂ - 2P₃)
#  𝗯 = -3P₁ - P₂ + 4P₃
#  C = P₁

struct QuadraticSegment{D, T}
    vertices::NTuple{3, Point{D, T}}
end

# -- Type aliases --

const QuadraticSegment2  = QuadraticSegment{2}
const QuadraticSegment2f = QuadraticSegment2{Float32}
const QuadraticSegment2d = QuadraticSegment2{Float64}

# -- Base --

Base.getindex(Q::QuadraticSegment, i::Integer) = Q.vertices[i]

# -- Constructors --

function QuadraticSegment(P1::Point{D, T}, P2::Point{D, T}, P3::Point{D, T}) where {D, T}
    return QuadraticSegment{D, T}((P1, P2, P3))
end

# -- Interpolation --

function quadratic_segment_weights(r)
    return ((2 * r - 1) * (r - 1),
            (2 * r - 1) *  r     ,
            -4 * r      * (r - 1))
end

function interpolate_quadratic_segment(P1::T, P2::T, P3::T, r) where {T}
    w = quadratic_segment_weights(r)
    return w[1] * P1 + w[2] * P2 + w[3] * P3
end

function interpolate_quadratic_segment(vertices::NTuple{3}, r)
    return mapreduce(*, +, quadratic_segment_weights(r), vertices)
end

function (Q::QuadraticSegment{D, T})(r::T) where {D, T}
    return interpolate_quadratic_segment(Q.vertices, r)
end

# -- Jacobian --

function quadratic_segment_jacobian(P1::T, P2::T, P3::T, r) where {T}
    return (4 * r - 3) * (P1 - P3) +
           (4 * r - 1) * (P2 - P3)
end

function quadratic_segment_jacobian(vertices::NTuple{3}, r)
    return (4 * r - 3) * (vertices[1] - vertices[3]) +
           (4 * r - 1) * (vertices[2] - vertices[3])
end

function jacobian(Q::QuadraticSegment{D, T}, r::T) where {D, T}
    return quadratic_segment_jacobian(Q.vertices, r)
end

# -- Measure --

function arclength(Q::QuadraticSegment)
    # The arc length integral may be reduced to an integral over the square root of a
    # quadratic polynomial using ‖𝘅‖ = √(𝘅 ⋅ 𝘅), which has an analytic solution.
    #              1             1
    # arc length = ∫ ‖Q′(r)‖dr = ∫ √(ar² + br + c) dr
    #              0             0
    #
    # If a = 0, we need to use a different formula, else the result is NaN.

    # Q(r) = r²𝗮 + 𝗯r + C,
    # where
    # 𝗮 = 2(P₁ + P₂ - 2P₃)
    # 𝗯 = -3P₁ - P₂ + 4P₃
    # C = P₁
    # hence,
    # Q'(r) = 2𝗮r + 𝗯,
    𝘃₁₃ = Q[3] - Q[1]
    𝘃₂₃ = Q[3] - Q[2]
    𝗮 = -2(𝘃₁₃ + 𝘃₂₃)
    # Move computation of 𝗯 to after exit.

    # ‖Q′(r)‖ =  √(4(𝗮 ⋅𝗮)r² + 4(𝗮 ⋅𝗯)r + 𝗯 ⋅𝗯) = √(ar² + br + c)
    # where
    # a = 4(𝗮 ⋅ 𝗮)
    # b = 4(𝗮 ⋅ 𝗯)
    # c = 𝗯 ⋅ 𝗯
    a = 4(𝗮 ⋅ 𝗮)
    # 0 ≤ a, since a = 4(𝗮 ⋅ 𝗮)  = 4 ‖𝗮‖², and 0 ≤ ‖𝗮‖²
    if a < 1e-5
        return distance(Q[1], Q[2])
    else

        𝗯 = 3𝘃₁₃ + 𝘃₂₃
        b = 4(𝗮 ⋅ 𝗯)
        c = 𝗯 ⋅ 𝗯

        # √(ar² + br + c) = √a √( (r + b₁)^2 + c₁)
        # where
        b₁ = b / (2 * a)
        c₁ = (c / a) - b₁^2
        #
        # Let u = r + b₁, then
        # 1                       1 + b₁
        # ∫ √(ar² + br + c) dr = √a ∫ √(u² + c₁) du
        # 0                         b₁
        #
        # This is an integral that exists in common integral tables.
        # Evaluation of the resultant expression may be simplified by using
        lb = b₁
        ub = 1 + b₁
        L = √(c₁ + lb^2)
        U = √(c₁ + ub^2)

        return √a * (U + lb * (U - L) + c₁ * ( atanh(ub / U) - atanh(lb / L) )) / 2
    end
end

# The area bounded by q and the line from P₁ to P₂ is 4/3 the area of the triangle
# formed by the vertices. Assumes the area is convex.
function area_enclosed_by(Q::QuadraticSegment2{T}) where {T}
    # Easily derived by transforming q such that P₁ = (0, 0) and P₂ = (x₂, 0).
    # However, vertices are CCW order, so sign of the area is flipped.
    return T(2 // 3) * ((Q[3] - Q[1]) × (Q[2] - Q[1]))
end

function area_enclosed_by_quadratic_segment(
        P1::Point2{T}, P2::Point2{T}, P3::Point2{T}) where {T}
    return T(2 // 3) * ((P3 - P1) × (P2 - P1))
end

# -- Centroid --

function centroid_of_area_enclosed_by(Q::QuadraticSegment2{T}) where {T}
    # For a quadratic segment, with P₁ = (0, 0), P₂ = (x₂, 0), and P₃ = (x₃, y₃),
    # where 0 < x₂, if the area bounded by q and the x-axis is convex, it can be
    # shown that the centroid of the area bounded by the segment and x-axis
    # is given by
    # C = (3x₂ + 4x₃, 4y₃) / 10
    #
    # To find the centroid of the area bounded by the segment for a general
    # quadratic segment, we transform the segment so that P₁ = (0, 0),
    # then use a change of basis (rotation) from the standard basis to the
    # following basis, to achieve y₂ = 0.
    #
    # Let v = (v₁, v₂) = (P₂ - P₁) / ‖P₂ - P₁‖
    # u₁ = ( v₁,  v₂) = v
    # u₂ = (-v₂,  v₁)
    #
    # Note: u₁ and u₂ are orthonormal.
    #
    # The transformation from the new basis to the standard basis is given by
    # U = [u₁ u₂] = | v₁ -v₂ |
    #               | v₂  v₁ |
    #
    # Since u₁ and u₂ are orthonormal, U is unitary.
    #
    # The transformation from the standard basis to the new basis is given by
    # U⁻¹ = Uᵗ = |  v₁  v₂ |
    #            | -v₂  v₁ |
    # since U is unitary.
    #
    # Therefore, the centroid of the area bounded by the segment is given by
    # C = U * Cᵤ + P₁
    # where
    # Cᵤ = (u₁ ⋅ (3(P₂ - P₁) + 4(P₃ - P₁)), 4(u₂ ⋅ (P₃ - P₁))) / 10
    v₁₂ = Q[2] - Q[1]
    four_v₁₃ = 4*(Q[3] - Q[1])
    u₁ = normalize(v₁₂)
    u₂ = Vec(-u₁[2], u₁[1])
    U  = Mat(u₁, u₂)
    Cᵤ = Vec(u₁ ⋅(3 * v₁₂ + four_v₁₃), u₂ ⋅ four_v₁₃) / 10
    return U * Cᵤ + Q[1]
end

function centroid_of_area_enclosed_by_quadratic_segment(
        P1::P, P2::P, P3::P) where {P <: Point2}
    v₁₂ = P2 - P1
    four_v₁₃ = 4*(P3 - P1)
    u₁ = normalize(v₁₂)
    u₂ = Vec(-u₁[2], u₁[1])
    U  = Mat(u₁, u₂)
    Cᵤ = Vec(u₁ ⋅(3 * v₁₂ + four_v₁₃), u₂ ⋅ four_v₁₃) / 10
    return U * Cᵤ + P1
end

# -- Bounding box --

function bounding_box(Q::QuadraticSegment2{T}) where {T}
    # Find the extrema for x and y by finding:
    # r_x such that dx/dr = 0    
    # r_y such that dy/dr = 0    
    # Q(r) = r²𝗮 + 𝗯r + C
    # Q′(r) = 2𝗮r + 𝗯 
    # (r_x, r_y) = -𝗯 ./ (2𝗮)    
    # Compare the extrema with the segment's endpoints to find the AABox    
    P1 = Q[1]
    P2 = Q[2]
    P3 = Q[3]
    𝘃₁₃ = P3 - P1
    𝘃₂₃ = P3 - P2
    a_x, a_y = -2(𝘃₁₃ + 𝘃₂₃)
    b_x, b_y = 3𝘃₁₃ + 𝘃₂₃
    r_x = b_x / (-2 * a_x)
    r_y = b_y / (-2 * a_y)
    xmin = min(P1[1], P2[1]); ymin = min(P1[2], P2[2])
    xmax = max(P1[1], P2[1]); ymax = max(P1[2], P2[2])
    if 0 < r_x < 1
        x_stationary = r_x * r_x * a_x + r_x * b_x + P1[1]
        xmin = min(xmin, x_stationary)
        xmax = max(xmax, x_stationary)
    end
    if 0 < r_y < 1
        y_stationary = r_y * r_y * a_y + r_y * b_y + P1[2]
        ymin = min(ymin, y_stationary)
        ymax = max(ymax, y_stationary)
    end
    return AABox(Point2(xmin, ymin), Point2(xmax, ymax))
end

# -- In --

function isleft(P::Point2{T}, Q::QuadraticSegment2{T}) where {T}
    # If the point is in the bounding box of the segment,
    # we need to check if the point is left of the segment.
    # To do this we must find the point on q that is closest to P.
    # At this Q(r) we compute Q'(r) × (P - Q(r)). If this quantity is
    # positive, then P is left of the segment.
    #
    # To compute Q_nearest, we find r which minimizes ‖P - Q(r)‖.
    # This r also minimizes ‖P - Q(r)‖².
    # It can be shown that this is equivalent to finding the minimum of the 
    # quartic function
    # ‖P - Q(r)‖² = f(r) = a₄r⁴ + a₃r³ + a₂r² + a₁r + a₀
    # The minimum of f(r) occurs when f′(r) = ar³ + br² + cr + d = 0, where
    # 𝘄 = P - P₁
    # a = 4(𝗮 ⋅ 𝗮)
    # b = 6(𝗮 ⋅ 𝗯)
    # c = 2[(𝗯  ⋅ 𝗯) - 2(𝗮 ⋅𝘄)]
    # d = -2(𝗯 ⋅ 𝘄)
    # Lagrange's method is used to find the roots.
    # (https://en.wikipedia.org/wiki/Cubic_equation#Lagrange's_method)    
    𝘃₁₃ = Q[3] - Q[1]
    𝘃₂₃ = Q[3] - Q[2]
    𝗮 = -2(𝘃₁₃ + 𝘃₂₃)    
    a = 4 * (𝗮 ⋅ 𝗮)

    if a < 1e-5 # quadratic is straight
        return 0 ≤ (Q[2] - Q[1]) × (P - Q[1])
    end

    𝗯 = 3𝘃₁₃ + 𝘃₂₃
    𝘄 = P - Q[1]

    b = 6 * (𝗮 ⋅ 𝗯)
    c = 2 * ((𝗯  ⋅ 𝗯) - 2 * (𝗮 ⋅𝘄))
    d = -2 * (𝗯 ⋅ 𝘄)

    # Lagrange's method
    e₁ = s₀ = -b / a
    e₂ = c / a
    e₃ = -d / a
    A = 2e₁^3 - 9e₁ * e₂ + 27e₃
    B = e₁^2 - 3e₂
    disc = A^2 - 4B^3
    if 0 < disc # one real root
        s₁ = ∛((A + √(disc)) / 2)
        if s₁ == 0
            s₂ = s₁
        else
            s₂ = B / s₁
        end
        r = (s₀ + s₁ + s₂) / 3
        return 0 ≤ jacobian(Q, r) × (P - Q(r))
    else # three real roots
        # t₁ is complex cube root
        t₁ = exp(log((A + √(complex(disc))) / 2) / 3)
        if t₁ == 0
            t₂ = t₁
        else
            t₂ = B / t₁
        end
        ζ₁ = Complex{T}(-1 / 2, √3 / 2)
        ζ₂ = conj(ζ₁)

        # Pick the point closest to P
        r = T(0)
        d = T(INF_POINT) 

        r1 = real((s₀ + t₁ + t₂)) / 3
        if 0 < r1 < 1
            d1 = distance2(P, Q(r1))
            if d1 < d
                r = r1
                d = d1
            end
        end

        r2 = real((s₀ + ζ₂ * t₁ + ζ₁ * t₂)) / 3
        if 0 < r2 < 1
            d2 = distance2(P, Q(r2))
            if d2 < d
                r = r2
                d = d2
            end
        end

        r3 = real((s₀ + ζ₁ * t₁ + ζ₂ * t₂)) / 3
        if 0 < r3 < 1
            d3 = distance2(P, Q(r3))
            if d3 < d
                r = r3
            end
        end

        return 0 ≤ jacobian(Q, r) × (P - Q(r))
    end
end

# -- IO --

function Base.show(io::IO, Q::QuadraticSegment{D, T}) where {D, T}
    type_char = '?'
    if T === Float32
        type_char = 'f'
    elseif T === Float64
        type_char = 'd'
    end
    print(io, "QuadraticSegment", D, type_char, '(',
        Q.vertices[1], ", ",
        Q.vertices[2], ", ",
        Q.vertices[3], ')')
end
