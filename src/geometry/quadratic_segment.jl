export QuadraticSegment,
       QuadraticSegment2,
       QuadraticSegment2f,
       QuadraticSegment2d

export interpolate_quadratic_segment

# QUADRATIC SEGMENT
# -----------------------------------------------------------------------------
#
# A quadratic segment represented by 3 vertices.
# These vertices are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#
# It is helpful to know:
#  q(r) = P₁ + r𝘂 + r²𝘃,
# where
#  𝘂 = 3𝘃₁₃ + 𝘃₂₃
#  𝘃 = -2(𝘃₁₃ + 𝘃₂₃)
# and
# 𝘃₁₃ = q[3] - q[1]
# 𝘃₂₃ = q[3] - q[2]
#
# NOTE: The equations above use 1-based indexing.
#

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
