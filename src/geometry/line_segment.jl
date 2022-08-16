export LineSegment,
       LineSegment2,
       LineSegment2f,
       LineSegment2d

export interpolate_line_segment,
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

struct LineSegment{D, T} <: AbstractEdge{D, T}
    vertices::Vec{2, Point{D, T}}
end

# -- Type aliases --

const LineSegment2     = LineSegment{2}
const LineSegment2f    = LineSegment2{Float32}
const LineSegment2d    = LineSegment2{Float64}

# -- Base --

Base.getindex(l::LineSegment, i) = l.vertices[i]
Base.broadcastable(l::LineSegment) = Ref(l)

# -- Constructors --

LineSegment(p1::Point{D, T}, p2::Point{D, T}) where {D, T} = LineSegment{D, T}(Vec(p1, p2))

# -- Interpolation --

function interpolate_line_segment(p1::T, p2::T, r) where {T}
    return p1 + r * (p2 - p1)
end

function interpolate_line_segment(vertices::Vec{2}, r)
    return vertices[1] + r * (vertices[2] - vertices[1])
end

function (l::LineSegment{D, T})(r::T) where {D, T}
    return interpolate_line_segment(l.vertices, r)
end

# -- Jacobian --

function line_segment_jacobian(p1::T, p2::T, r) where {T}
    return p2 - p1
end

function line_segment_jacobian(vertices::Vec{2}, r)
    return vertices[2] - vertices[1]
end

function jacobian(l::LineSegment{D, T}, r::T) where {D, T}
    return line_segment_jacobian(l.vertices, r)
end

# -- Measure --

arclength(l::LineSegment) = distance(l[1], l[2])

# -- Bounding box --

function bounding_box(l::LineSegment{2, T}) where {T}
    return bounding_box(l.vertices)
end

# -- In --

isleft(P::Point{2}, l::LineSegment{2}) = 0 ≤ (l[2] - l[1]) × (P - l[1])

# -- IO --

function Base.show(io::IO, l::LineSegment{D, T}) where {D, T}
    type_char = '?'
    if T === Float32
        type_char = 'f'
    elseif T === Float64
        type_char = 'd'
    end
    print(io, "LineSegment", D, type_char, '(',
        l.vertices[1], ", ",
        l.vertices[2], ')')
end
