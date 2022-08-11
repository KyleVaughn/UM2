export LineSegment,
       LineSegment2,
       LineSegment2f,
       LineSegment2d

export interpolate_line_segment,
       jacobian_line_segment,
       jacobian

# LINE SEGMENT
# -----------------------------------------------------------------------------
#
# A line segment represented by its two vertices.
# These vertices are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#

struct LineSegment{D, T} <: Edge{D, T}
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

function interpolate_line_segment(vertices::Vec, r)
    return vertices[1] + r * (vertices[2] - vertices[1])
end

function (l::LineSegment{D, T})(r::T) where {D, T}
    return interpolate_line_segment(l.vertices, r)
end

# -- Jacobian --

function jacobian_line_segment(p1::T, p2::T, r) where {T}    
    return p2 - p1    
end

function jacobian_line_segment(vertices::Vec, r)
    return vertices[2] - vertices[1]
end

function jacobian(l::LineSegment{D, T}, r::T) where {D, T}
    return jacobian_line_segment(l.vertices, r)
end

# -- IO --

function Base.show(io::IO, l::LineSegment{D, T}) where {D, T}
    print(io, "LineSegment", D) 
    if T === Float32
        print(io, 'f')
    elseif T === Float64
        print(io, 'd')
    else
        print(io, '?')
    end
    print('(', l.vertices[1], ", ", l.vertices[2], ")")
end
