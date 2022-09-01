export Quadrilateral,
       Quadrilateral2,
       Quadrilateral2f,
       Quadrilateral2d

export interpolate_quadrilateral,
       jacobian, quadrilateral_jacobian,
       area, quadrilateral_area,
       centroid, quadrilateral_centroid,
       triangulate

# QUADRILATERAL
# -----------------------------------------------------------------------------
#
# A quadrilateral represented by its 4 vertices.
# These vertices are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#

const Quadrilateral = Polygon{4}

# -- Type aliases --

const Quadrilateral2  = Quadrilateral{2}
const Quadrilateral2f = Quadrilateral2{f32}
const Quadrilateral2d = Quadrilateral2{f64}

# -- Constructors --

function Quadrilateral(
        P1::Point{D, T},
        P2::Point{D, T},
        P3::Point{D, T},
        P4::Point{D, T}) where {D, T}
    return Quadrilateral{D, T}((P1, P2, P3, P4))
end

# -- Interpolation --

function quadrilateral_weights(r, s)
    return ((1 - r) * (1 - s),
                 r  * (1 - s),
                 r  *      s ,
            (1 - r) *      s )
end

# Assumes a convex quadrilateral
function interpolate_quadrilateral(P1::T, P2::T, P3::T, P4::T, r, s) where {T}
    w = quadrilateral_weights(r, s)
    return w[1] * P1 + w[2] * P2 + w[3] * P3 + w[4] * P4
end

function interpolate_quadrilateral(vertices::NTuple{4}, r, s)
    return mapreduce(*, +, quadrilateral_weights(r, s), vertices)
end

function (Q::Quadrilateral{D, T})(r::T, s::T) where {D, T}
    return interpolate_quadrilateral(Q.vertices, r, s)
end

# -- Jacobian --

# Assumes a convex quadrilateral
function quadrilateral_jacobian(P1::T, P2::T, P3::T, P4::T, r, s) where {T}
    ∂r = (1 - s) * (P2 - P1) - s * (P4 - P3)
    ∂s = (1 - r) * (P4 - P1) - r * (P2 - P3)
    return Mat(∂r, ∂s)
end

function quadrilateral_jacobian(vertices::NTuple{4}, r, s)
    ∂r = (1 - s) * (vertices[2] - vertices[1]) - s * (vertices[4] - vertices[3])
    ∂s = (1 - r) * (vertices[4] - vertices[1]) - r * (vertices[2] - vertices[3])
    return Mat(∂r, ∂s)
end

function jacobian(Q::Quadrilateral{D, T}, r::T, s::T) where {D, T}
    return quadrilateral_jacobian(Q.vertices, r, s)
end

# -- Measure --

# Assumes a convex quadrilateral
function area(Q::Quadrilateral2)
    return ((Q[3] - Q[1]) × (Q[4] - Q[2])) / 2
end

function quadrilateral_area(P1::P, P2::P, P3::P, P4::P) where {P <: Point2}
    return ((P3 - P1) × (P4 - P2)) / 2
end

# -- Centroid --

# Assumes a convex quadrilateral
function centroid(Q::Quadrilateral2)
    # By geometric decomposition into two triangles
    v₁₂ = Q[2] - Q[1]
    v₁₃ = Q[3] - Q[1]
    v₁₄ = Q[4] - Q[1]
    a₁ = v₁₂ × v₁₃
    ma₂ = v₁₄ × v₁₃ # minus a₂. Flipped for potential SSE optimization
    P₁₃ = Q[1] + Q[3]
    return (a₁ * (P₁₃ + q[2]) - ma₂ * (P₁₃ + q[4])) / (3 * (a₁ - ma₂))
end

function quadrilateral_centroid(P1::P, P2::P, P3::P, P4::P) where {P <: Point2}
    v₁₂ = P2 - P1
    v₁₃ = P3 - P1
    v₁₄ = P4 - P1
    a₁ = v₁₂ × v₁₃
    ma₂ = v₁₄ × v₁₃ # minus a₂. Flipped for potential SSE optimization
    P₁₃ = P1 + P3
    return (a₁ * (P₁₃ + P2) - ma₂ * (P₁₃ + P4)) / (3 * (a₁ - ma₂))
end

# -- Triangulation --

# Assumes a convex quadrilateral
function triangulate(Q::Quadrilateral2)
    return (
        Triangle(Q[1], Q[2], Q[3]),
        Triangle(Q[1], Q[3], Q[4])
       )
end

# -- IO --

function Base.show(io::IO, Q::Quadrilateral{D, T}) where {D, T}
    type_char = '?'                                        
    if T === Float32
        type_char = 'f'
    elseif T === Float64
        type_char = 'd'
    end
    print(io, "Quadrilateral", D, type_char, '(', 
        Q.vertices[1], ", ", 
        Q.vertices[2], ", ", 
        Q.vertices[3], ", ",
        Q.vertices[4], ')')
end
