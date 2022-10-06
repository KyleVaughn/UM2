export LineSegment,
       LineSegment2,
       LineSegment2f,
       LineSegment2d

export vertices,
       interpolate_line_segment,
       jacobian,
       line_segment_jacobian,
       arclength,
       bounding_box,
       isleft

# LINE SEGMENT
# -----------------------------------------------------------------------------
#
# A line segment represented by its two vertices.
# These vertices are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#

struct LineSegment{D, T}
    vertices::NTuple{2, Point{D, T}}
end

# -- Type aliases --

const LineSegment2  = LineSegment{2}
const LineSegment2f = LineSegment2{Float32}
const LineSegment2d = LineSegment2{Float64}

# -- Base --

Base.getindex(L::LineSegment, i::Integer) = L.vertices[i]

# -- Accessors --

vertices(l::LineSegment) = l.vertices

# -- Constructors --

LineSegment(P1::Point{D, T}, P2::Point{D, T}) where {D, T} = LineSegment{D, T}((P1, P2))

# -- Interpolation --

function interpolate_line_segment(P1::T, P2::T, r) where {T}
    return P1 + r * (P2 - P1)
end

function interpolate_line_segment(vertices::NTuple{2}, r)
    return vertices[1] + r * (vertices[2] - vertices[1])
end

function (L::LineSegment{D, T})(r::T) where {D, T}
    return interpolate_line_segment(L.vertices, r)
end

# -- Jacobian --

function line_segment_jacobian(P1::T, P2::T, r) where {T}
    return P2 - P1
end

function line_segment_jacobian(vertices::NTuple{2}, r)
    return vertices[2] - vertices[1]
end

function jacobian(L::LineSegment{D, T}, r::T) where {D, T}
    return line_segment_jacobian(L.vertices, r)
end

# -- Measure --

arclength(L::LineSegment) = distance(L[1], L[2])

# -- Bounding box --

bounding_box(L::LineSegment) = bounding_box(L.vertices)

# -- In --

isleft(P::Point{2}, L::LineSegment{2}) = 0 ≤ (L[2] - L[1]) × (P - L[1])

# -- IO --

function Base.show(io::IO, L::LineSegment{D, T}) where {D, T}
    type_char = '?'
    if T === Float32
        type_char = 'f'
    elseif T === Float64
        type_char = 'd'
    end
    print(io, "LineSegment", D, type_char, '(',
        L.vertices[1], ", ",
        L.vertices[2], ')')
end
