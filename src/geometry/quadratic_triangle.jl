export QuadraticTriangle,
       QuadraticTriangle2,
       QuadraticTriangle2f,
       QuadraticTriangle2d

export interpolate_quadratic_triangle

# QUADRATIC TRIANGLE 
# -----------------------------------------------------------------------------
#
# A quadratic triangle represented by 6 vertices.
# These vertices are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#

struct QuadraticTriangle{D, T} <: Polygon{D, T}
    vertices::Vec{6, Point{D, T}}
end

# -- Type aliases --

const QuadraticTriangle2  = QuadraticTriangle{2}
const QuadraticTriangle2f = QuadraticTriangle2{Float32}
const QuadraticTriangle2d = QuadraticTriangle2{Float64}

# -- Base --

Base.getindex(t::QuadraticTriangle, i) = t.vertices[i]
Base.broadcastable(t::QuadraticTriangle) = Ref(t)

# -- Constructors --

function QuadraticTriangle(
        p1::Point{D, T}, 
        p2::Point{D, T}, 
        p3::Point{D, T},
        p4::Point{D, T},
        p5::Point{D, T},
        p6::Point{D, T}) where {D, T}
    return QuadraticTriangle{D, T}(Vec(p1, p2, p3, p4, p5, p6))
end

# -- Interpolation --

function interpolate_quadratic_triangle(
        p1::T, p2::T, p3::T, p4::T, p5::T, p6::T, r) where {T}
    return ((2 * (1 - r - s) - 1) * ( 1 - r - s)) * p1 +    
           (          r           * ( 2 * r - 1)) * p2 +    
           (              s       * ( 2 * s - 1)) * p3 +    
           (      4 * r           * ( 1 - r - s)) * p4 +    
           (      4 * r           *           s ) * p5 +    
           (          4 * s       * ( 1 - r - s)) * p6
end

function interpolate_quadratic_triangle(vertices::Vec, r)
    return ((2 * (1 - r - s) - 1) * ( 1 - r - s)) * vertices[1] +    
           (          r           * ( 2 * r - 1)) * vertices[2] +    
           (              s       * ( 2 * s - 1)) * vertices[3] +    
           (      4 * r           * ( 1 - r - s)) * vertices[4] +    
           (      4 * r           *           s ) * vertices[5] +    
           (          4 * s       * ( 1 - r - s)) * vertices[6]
end

function (t::QuadraticTriangle{D, T})(r::T) where {D, T}
    return interpolate_quadratic_triangle(t.vertices, r)
end

# -- IO --

function Base.show(io::IO, t::QuadraticTriangle{D, T}) where {D, T}
    print(io, "QuadraticTriangle", D) 
    if T === Float32
        print(io, 'f')
    elseif T === Float64
        print(io, 'd')
    else
        print(io, '?')
    end
    print('(', t.vertices[1], ", ", 
               t.vertices[2], ", ",
               t.vertices[3], ", ",
               t.vertices[4], ", ",
               t.vertices[5], ", ",
               t.vertices[6], ")")
end
