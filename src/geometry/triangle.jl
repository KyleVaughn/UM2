export Triangle,
       Triangle2,
       Triangle2f,
       Triangle2d

export interpolate_triangle,
       jacobian,
       triangle_jacobian,
       area,
       triangle_area,
       centroid,
       triangle_centroid,
       edge,
       edges,
       bounding_box

# TRIANGLE
# -----------------------------------------------------------------------------
#
# A triangle represented by its 3 vertices.
# These vertices are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#

struct Triangle{D, T} <: Polygon{D, T}
    vertices::Vec{3, Point{D, T}}
end

# -- Type aliases --

const Triangle2  = Triangle{2}
const Triangle2f = Triangle2{Float32}
const Triangle2d = Triangle2{Float64}

# -- Base --

Base.getindex(t::Triangle, i) = t.vertices[i]
Base.broadcastable(t::Triangle) = Ref(t)

# -- Constructors --

function Triangle(p1::Point{D, T}, p2::Point{D, T}, p3::Point{D, T}) where {D, T}
    return Triangle{D, T}(Vec(p1, p2, p3))
end

# -- Interpolation --

function interpolate_triangle(p1::T, p2::T, p3::T, r, s) where {T}
    return (1 - r - s) * p1 + r * p2 + s * p3
end

function interpolate_triangle(vertices::Vec, r, s)
    return (1 - r - s) * vertices[1] + r * vertices[2] + s * vertices[3]
end

function (t::Triangle{D, T})(r::T, s::T) where {D, T}
    return interpolate_triangle(t.vertices, r, s)
end

# -- Jacobian --

function triangle_jacobian(p1::T, p2::T, p3::T, r, s) where {T}
    ∂r = p2 - p1
    ∂s = p3 - p1
    return Mat(∂r, ∂s)
end

function triangle_jacobian(vertices::Vec{3}, r, s)
    ∂r = vertices[2] - vertices[1]
    ∂s = vertices[3] - vertices[1]
    return Mat(∂r, ∂s)
end

function jacobian(t::Triangle{D, T}, r::T, s::T) where {D, T}
    return triangle_jacobian(t.vertices, r, s)
end

# -- Measure --

area(t::Triangle{2}) = ((t[2] - t[1]) × (t[3] - t[1])) / 2
area(t::Triangle{3}) = norm((t[2] - t[1]) × (t[3] - t[1])) / 2

function triangle_area(p1::P, p2::P, p3::P) where {P <: Point{2}}
    return ((p2 - p1) × (p3 - p1))/ 2
end

# -- Centroid --

centroid(t::Triangle) = (t[1] + t[2] + t[3]) / 3

function triangle_centroid(p1::P, p2::P, p3::P) where {P <: Point{2}}
    return (p1 + p2 + p3) / 3
end

# -- Edges --

function edge(i::Integer, t::Triangle)
    # Assumes 1 ≤ i ≤ 3.
    if i < 3
        return LineSegment(t[i], t[i+1])
    else
        return LineSegment(t[3], t[1])
    end
end

edges(t::Triangle) = (edge(i, t) for i in 1:3)

# -- Bounding box --

function bounding_box(t::Triangle)
    return bounding_box(t.vertices)
end

# -- In --

Base.in(P::Point{2}, t::Triangle{2}) = all(edge -> isleft(P, edge), edges(t))

# -- IO --

function Base.show(io::IO, t::Triangle{D, T}) where {D, T}
    type_char = '?'
    if T === Float32
        type_char = 'f'
    elseif T === Float64
        type_char = 'd'
    end
    print(io, "Triangle", D, type_char, '(', 
        t.vertices[1], ", ", 
        t.vertices[2], ", ", 
        t.vertices[3], ')')
end
