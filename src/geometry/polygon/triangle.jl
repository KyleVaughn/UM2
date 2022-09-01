export Triangle,
       Triangle2,
       Triangle2f,
       Triangle2d

export interpolate_triangle,
       jacobian, triangle_jacobian,
       area, triangle_area,
       centroid, triangle_centroid

# TRIANGLE
# -----------------------------------------------------------------------------
#
# A triangle represented by its 3 vertices.
# These vertices are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#

const Triangle = Polygon{3}

# -- Type aliases --

const Triangle2  = Triangle{2}
const Triangle2f = Triangle2{Float32}
const Triangle2d = Triangle2{Float64}

# -- Constructors --

function Triangle(P1::Point{D, T}, P2::Point{D, T}, P3::Point{D, T}) where {D, T}
    return Triangle{D, T}((P1, P2, P3))
end

# -- Interpolation --

function triangle_weights(r, s)
    return (1 - r - s, r, s)
end

function interpolate_triangle(P1::T, P2::T, P3::T, r, s) where {T}
    w = triangle_weights(r, s)
    return w[1] * P1 + w[2] * P2 + w[3] * P3
end

function interpolate_triangle(vertices::NTuple{3}, r, s)
    return mapreduce(*, +, triangle_weights(r, s), vertices)
end

function (t::Triangle{D, T})(r::T, s::T) where {D, T}
    return interpolate_triangle(t.vertices, r, s)
end

# -- Jacobian --

function triangle_jacobian(P1::T, P2::T, P3::T, r, s) where {T}
    ∂r = P2 - P1
    ∂s = P3 - P1
    return Mat(∂r, ∂s)
end

function triangle_jacobian(vertices::NTuple{3}, r, s)
    ∂r = vertices[2] - vertices[1]
    ∂s = vertices[3] - vertices[1]
    return Mat(∂r, ∂s)
end

function jacobian(T::Triangle{D, F}, r::F, s::F) where {D, F}
    return triangle_jacobian(T.vertices, r, s)
end

# -- Measure --

area(T::Triangle2) = ((T[2] - T[1]) × (T[3] - T[1])) / 2
#area(T::Triangle3) = norm((T[2] - t[1]) × (t[3] - t[1])) / 2

function triangle_area(P1::P, P2::P, P3::P) where {P <: Point2}
    return ((P2 - P1) × (P3 - P1))/ 2
end

# -- Centroid --

centroid(T::Triangle) = (T[1] + T[2] + T[3]) / 3

function triangle_centroid(P1::P, P2::P, P3::P) where {P <: Point2}
    return (P1 + P2 + P3) / 3
end

# -- In --

function Base.in(P::Point2, T::Triangle2)
    return isCCW(T[1], T[2], P) && 
           isCCW(T[2], T[3], P) && 
           isCCW(T[3], T[1], P)
end

# -- IO --

function Base.show(io::IO, T::Triangle{D, F}) where {D, F}
    type_char = '?'
    if F === Float32
        type_char = 'f'
    elseif F === Float64
        type_char = 'd'
    end
    print(io, "Triangle", D, type_char, '(', 
        T.vertices[1], ", ", 
        T.vertices[2], ", ", 
        T.vertices[3], ')')
end
