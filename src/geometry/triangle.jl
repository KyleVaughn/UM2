export Triangle,
       Triangle2,
       Triangle2f,
       Triangle2d

export vertices,
       interpolate_triangle,
       jacobian,
       triangle_jacobian,
       area,
       triangle_area,
       centroid,
       triangle_centroid,
       edge,
       edge_iterator,
       bounding_box

# TRIANGLE
# -----------------------------------------------------------------------------
#
# A triangle represented by its 3 vertices.
# These vertices are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#

struct Triangle{D, T}
    vertices::Vec{3, Point{D, T}}
end

# -- Type aliases --

const Triangle2  = Triangle{2}
const Triangle2f = Triangle2{Float32}
const Triangle2d = Triangle2{Float64}

# -- Base --

Base.getindex(T::Triangle, i) = T.vertices[i]
Base.broadcastable(T::Triangle) = Ref(t)

# -- Accessors --

vertices(T::Triangle) = T.vertices

# -- Constructors --

function Triangle(P1::Point{D, T}, P2::Point{D, T}, P3::Point{D, T}) where {D, T}
    return Triangle{D, T}(Vec(P1, P2, P3))
end

# -- Interpolation --

function interpolate_triangle(P1::T, P2::T, P3::T, r, s) where {T}
    return (1 - r - s) * P1 + r * P2 + s * P3
end

function interpolate_triangle(vertices::Vec, r, s)
    return (1 - r - s) * vertices[1] + r * vertices[2] + s * vertices[3]
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

function triangle_jacobian(vertices::Vec{3}, r, s)
    ∂r = vertices[2] - vertices[1]
    ∂s = vertices[3] - vertices[1]
    return Mat(∂r, ∂s)
end

function jacobian(T::Triangle{D, F}, r::F, s::F) where {D, F}
    return triangle_jacobian(T.vertices, r, s)
end

# -- Measure --

area(T::Triangle{2}) = ((T[2] - T[1]) × (T[3] - T[1])) / 2
area(T::Triangle{3}) = norm((T[2] - t[1]) × (t[3] - t[1])) / 2

function triangle_area(P1::P, P2::P, P3::P) where {P <: Point{2}}
    return ((P2 - P1) × (P3 - P1))/ 2
end

# -- Centroid --

centroid(T::Triangle) = (T[1] + T[2] + T[3]) / 3

function triangle_centroid(P1::P, P2::P, P3::P) where {P <: Point{2}}
    return (P1 + P2 + P3) / 3
end

# -- Edges --

function ev_conn(i::Integer, fv_conn::NTuple{3, I}) where {I <: Integer}
    # Assumes 1 ≤ i ≤ 3.
    if i < 3
        return (fv_conn[i], fv_conn[i + 1])
    else
        return (fv_conn[3], fv_conn[1])
    end
end

function ev_conn_iterator(fv_conn::NTuple{3, I}) where {I}
    return (ev_conn(i, fv_conn) for i in 1:3)
end

function edge(i::Integer, T::Triangle)
    # Assumes 1 ≤ i ≤ 3.
    if i < 3
        return LineSegment(T[i], T[i+1])
    else
        return LineSegment(T[3], T[1])
    end
end

edge_iterator(T::Triangle) = (edge(i, T) for i in 1:3)

# -- Bounding box --

function bounding_box(T::Triangle)
    return bounding_box(T.vertices)
end

# -- In --

Base.in(P::Point{2}, T::Triangle{2}) = all(edge -> isleft(P, edge), edge_iterator(T))

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
