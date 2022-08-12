export Quadrilateral,
       Quadrilateral2,
       Quadrilateral2f,
       Quadrilateral2d

export interpolate_quadrilateral,
       jacobian_quadrilateral,
       jacobian,
       area

# QUADRILATERAL
# -----------------------------------------------------------------------------
#
# A quadrilateral represented by its 4 vertices.
# These vertices are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#

struct Quadrilateral{D, T} <: Polygon{D, T}
    vertices::Vec{4, Point{D, T}}
end

# -- Type aliases --

const Quadrilateral2  = Quadrilateral{2}
const Quadrilateral2f = Quadrilateral2{Float32}
const Quadrilateral2d = Quadrilateral2{Float64}

# -- Base --

Base.getindex(t::Quadrilateral, i) = t.vertices[i]
Base.broadcastable(t::Quadrilateral) = Ref(t)

# -- Constructors --

function Quadrilateral(
        p1::Point{D, T}, 
        p2::Point{D, T}, 
        p3::Point{D, T},
        p4::Point{D, T}) where {D, T}
    return Quadrilateral{D, T}(Vec(p1, p2, p3, p4))
end

# -- Interpolation --

function interpolate_quadrilateral(p1::T, p2::T, p3::T, p4::T, r, s) where {T}
    return ((1 - r) * (1 - s)) * p1 +    
           (     r  * (1 - s)) * p2 +    
           (     r  *      s ) * p3 +    
           ((1 - r) *      s ) * p4
end

function interpolate_quadrilateral(vertices::Vec{4}, r, s)
    return ((1 - r) * (1 - s)) * vertices[1] +    
           (     r  * (1 - s)) * vertices[2] +    
           (     r  *      s ) * vertices[3] +    
           ((1 - r) *      s ) * vertices[4]
end

function (q::Quadrilateral{D, T})(r::T) where {D, T}
    return interpolate_quadrilateral(q.vertices, r, s)
end

# -- Jacobian --

function jacobian_quadrilateral(p1::T, p2::T, p3::T, p4::T, r, s) where {T}
    ∂r = (1 - s) * (p2 - p1) - s * (p4 - p3)
    ∂s = (1 - r) * (p4 - p1) - r * (p2 - p3)
    return Mat(∂r, ∂s)
end

function jacobian_quadrilateral(vertices::Vec{4}, r, s)
    ∂r = (1 - s) * (vertices[2] - vertices[1]) - s * (vertices[4] - vertices[3])
    ∂s = (1 - r) * (vertices[4] - vertices[1]) - r * (vertices[2] - vertices[3])
    return Mat(∂r, ∂s)
end

function jacobian(q::Quadrilateral{D, T}, r::T, s::T) where {D, T}
    return jacobian_quadrilateral(q.vertices, r, s)
end

# -- Measure --

# Assumes convex quadrilateral, so fan triangulation can be used.
function area(q::Quadrilateral{2})
    return ((q[2] - q[1]) × (q[3] - q[1]) -
            (q[4] - q[1]) × (q[3] - q[1]))  / 2
end

# -- IO --

function Base.show(io::IO, q::Quadrilateral{D, T}) where {D, T}
    print(io, "Quadrilateral", D) 
    if T === Float32
        print(io, 'f')
    elseif T === Float64
        print(io, 'd')
    else
        print(io, '?')
    end
    print('(', q.vertices[1], ", ", 
               q.vertices[2], ", ", 
               q.vertices[3], ", ",
               q.vertices[4], ")")
end
