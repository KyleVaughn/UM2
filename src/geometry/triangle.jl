export Triangle,
       Triangle2,
       Triangle2f,
       Triangle2d

export interpolate_triangle,
       jacobian_triangle,
       jacobian,
       area,
       triangle_area

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

function jacobian_triangle(p1::T, p2::T, p3::T, r, s) where {T}
    ∂r = p2 - p1
    ∂s = p3 - p1
    return Mat(∂r, ∂s)
end

function jacobian_triangle(vertices::Vec{3}, r, s)
    ∂r = vertices[2] - vertices[1]
    ∂s = vertices[3] - vertices[1]
    return Mat(∂r, ∂s)
end

function jacobian(t::Triangle{D, T}, r::T, s::T) where {D, T}
    return jacobian_triangle(t.vertices, r, s)
end

# -- Measure --

area(t::Triangle{2}) = ((t[2] - t[1]) × (t[3] - t[1])) / 2
area(t::Triangle{3}) = norm((t[2] - t[1]) × (t[3] - t[1])) / 2
function triangle_area(p1::P, p2::P, p3::P) where {P <: Point{2}}
    return ((p2 - p1) × (p3 - p1))/ 2
end

# -- IO --

function Base.show(io::IO, t::Triangle{D, T}) where {D, T}
    print(io, "Triangle", D)
    if T === Float32
        print(io, 'f')
    elseif T === Float64
        print(io, 'd')
    else
        print(io, '?')
    end
    print('(', t.vertices[1], ", ", t.vertices[2], ", ", t.vertices[3], ")")
end
