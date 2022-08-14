export QuadraticSegment,
       QuadraticSegment2,
       QuadraticSegment2f,
       QuadraticSegment2d

export interpolate_quadratic_segment,
       jacobian_quadratic_segment,
       jacobian,
       arclength,
       area_enclosed_by,
       enclosed_area_quadratic_segment

# QUADRATIC SEGMENT
# -----------------------------------------------------------------------------
#
# A quadratic segment represented by 3 vertices.
# These vertices are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#
# It is helpful to know:
#  q(r) = r²𝗮 + 𝗯r + 𝗰,
# where
#  𝗮 = 2(P₁ + P₂ - 2P₃)
#  𝗯 = -3P₁ - P₂ + 4P₃
#  𝗰 = P₁

struct QuadraticSegment{D, T} <: Edge{D, T}
    vertices::Vec{3, Point{D, T}}
end

# -- Type aliases --

const QuadraticSegment2  = QuadraticSegment{2}
const QuadraticSegment2f = QuadraticSegment2{Float32}
const QuadraticSegment2d = QuadraticSegment2{Float64}

# -- Base --

Base.getindex(q::QuadraticSegment, i) = q.vertices[i]
Base.broadcastable(q::QuadraticSegment) = Ref(q)

# -- Constructors --

function QuadraticSegment(p1::Point{D, T}, p2::Point{D, T}, p3::Point{D, T}) where {D, T}
    return QuadraticSegment{D, T}(Vec(p1, p2, p3))
end

# -- Interpolation --

function interpolate_quadratic_segment(p1::T, p2::T, p3::T, r) where {T}
    return ((2 * r - 1) * (r - 1)) * p1 +
           ((2 * r - 1) *  r     ) * p2 +
           (-4 * r      * (r - 1)) * p3
end

function interpolate_quadratic_segment(vertices::Vec, r)
    return ((2 * r - 1) * (r - 1)) * vertices[1] +
           ((2 * r - 1) *  r     ) * vertices[2] +
           (-4 * r      * (r - 1)) * vertices[3]
end

function (q::QuadraticSegment{D, T})(r::T) where {D, T}
    return interpolate_quadratic_segment(q.vertices, r)
end

# -- Jacobian --

function jacobian_quadratic_segment(p1::T, p2::T, p3::T, r) where {T}
    return (4 * r - 3) * (p1 - p3) +
           (4 * r - 1) * (p2 - p3)
end

function jacobian_quadratic_segment(vertices::Vec{3}, r)
    return (4 * r - 3) * (vertices[1] - vertices[3]) +
           (4 * r - 1) * (vertices[2] - vertices[3])
end

function jacobian(q::QuadraticSegment{D, T}, r::T) where {D, T}
    return jacobian_quadratic_segment(q.vertices, r)
end

# -- Measure --

function arclength(q::QuadraticSegment)
    # The arc length integral may be reduced to an integral over the square root of a
    # quadratic polynomial using ‖𝘅‖ = √(𝘅 ⋅ 𝘅), which has an analytic solution.
    #              1             1
    # arc length = ∫ ‖q′(r)‖dr = ∫ √(ar² + br + c) dr
    #              0             0
    #
    # If a = 0, we need to use a different formula.
    
    # q(r) = r²𝗮 + 𝗯r + 𝗰,
    # where
    # 𝗮 = 2(P₁ + P₂ - 2P₃)
    # 𝗯 = -3P₁ - P₂ + 4P₃
    # 𝗰 = P₁
    # hence,
    # q'(r) = 2𝗮r + 𝗯,
    𝘃₁₃ = q[3] - q[1]
    𝘃₂₃ = q[3] - q[2]
    𝗮 = -2(𝘃₁₃ + 𝘃₂₃)
    𝗯 = 3𝘃₁₃ + 𝘃₂₃

    # ‖q′(r)‖ =  √(4(𝗮 ⋅𝗮)r² + 4(𝗮 ⋅𝗯)r + 𝗯 ⋅𝗯) = √(ar² + br + c)
    # where
    # a = 4(𝗮 ⋅ 𝗮)
    # b = 4(𝗮 ⋅ 𝗯)
    # c = 𝗯 ⋅ 𝗯
    a = 4(𝗮 ⋅ 𝗮)
    # 0 ≤ a, since a = 4(𝗮 ⋅ 𝗮)  = 4 ‖𝗮‖², and 0 ≤ ‖𝗮‖²
    if a < 1e-5 
        return distance(q[1], q[2])
    else
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

# The area bounded by q is 4/3 the area of the triangle formed by the vertices.
function area_enclosed_by(q::QuadraticSegment{2, T}) where {T}
    return T(2 // 3) * (q[2] - q[1]) × (q[3] - q[1])
end

function enclosed_area_quadratic_segment(p1::P, p2::P, p3::P) where {P <: Point{2}}
    return T(2 // 3) * (p2 - p1) × (p3 - p1)
end

# -- IO --

function Base.show(io::IO, q::QuadraticSegment{D, T}) where {D, T}
    print(io, "QuadraticSegment", D)
    if T === Float32
        print(io, 'f')
    elseif T === Float64
        print(io, 'd')
    else
        print(io, '?')
    end
    print('(', q.vertices[1], ", ", q.vertices[2], ", ", q.vertices[3], ")")
end
