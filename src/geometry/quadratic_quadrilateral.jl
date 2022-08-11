export QuadraticQuadrilateral,
       QuadraticQuadrilateral2,
       QuadraticQuadrilateral2f,
       QuadraticQuadrilateral2d

export interpolate_quadratic_quadrilateral

# QUADRATIC QUADRILATERAL
# -----------------------------------------------------------------------------
#
# A quadratic quadrilateral represented by 8 vertices.
# These vertices are D-dimensional points of type T.
#
# See chapter 8 of the VTK book for more info.
#

struct QuadraticQuadrilateral{D, T} <: Polygon{D, T}
    vertices::Vec{8, Point{D, T}}
end

# -- Type aliases --

const QuadraticQuadrilateral2  = QuadraticQuadrilateral{2}
const QuadraticQuadrilateral2f = QuadraticQuadrilateral2{Float32}
const QuadraticQuadrilateral2d = QuadraticQuadrilateral2{Float64}

# -- Base --

Base.getindex(t::QuadraticQuadrilateral, i) = t.vertices[i]
Base.broadcastable(t::QuadraticQuadrilateral) = Ref(t)

# -- Constructors --

function QuadraticQuadrilateral(
        p1::Point{D, T}, 
        p2::Point{D, T}, 
        p3::Point{D, T},
        p4::Point{D, T},
        p5::Point{D, T},
        p6::Point{D, T},
        p7::Point{D, T},
        p8::Point{D, T}) where {D, T}
    return QuadraticQuadrilateral{D, T}(Vec(p1, p2, p3, p4, p5, p6, p7, p8))
end

# -- Interpolation --

function interpolate_quadratic_quadrilateral(p1::T, p2::T, p3::T, p4::T, 
                                             p5::T, p6::T, p7::T, p8::T, r) where {T}
    xi = 2 * r - 1;    
    eta = 2 * s - 1;    
    return ((1 - xi) * (1 - eta) * (-xi - eta - 1) / 4) * p1 +    
           ((1 + xi) * (1 - eta) * ( xi - eta - 1) / 4) * p2 +    
           ((1 + xi) * (1 + eta) * ( xi + eta - 1) / 4) * p3 +    
           ((1 - xi) * (1 + eta) * (-xi + eta - 1) / 4) * p4 +    
                      ((1 -  xi *  xi) * (1 - eta) / 2) * p5 +    
                      ((1 - eta * eta) * (1 +  xi) / 2) * p6 +    
                      ((1 -  xi *  xi) * (1 + eta) / 2) * p7 +    
                      ((1 - eta * eta) * (1 -  xi) / 2) * p8
end

function interpolate_quadratic_quadrilateral(vertices::Vec, r)
    xi = 2 * r - 1;
    eta = 2 * s - 1;
    return ((1 - xi) * (1 - eta) * (-xi - eta - 1) / 4) * vertices[1] +
           ((1 + xi) * (1 - eta) * ( xi - eta - 1) / 4) * vertices[2] +
           ((1 + xi) * (1 + eta) * ( xi + eta - 1) / 4) * vertices[3] +
           ((1 - xi) * (1 + eta) * (-xi + eta - 1) / 4) * vertices[4] +
                      ((1 -  xi *  xi) * (1 - eta) / 2) * vertices[5] +
                      ((1 - eta * eta) * (1 +  xi) / 2) * vertices[6] +
                      ((1 -  xi *  xi) * (1 + eta) / 2) * vertices[7] +
                      ((1 - eta * eta) * (1 -  xi) / 2) * vertices[8]
end

function (q::QuadraticQuadrilateral{D, T})(r::T) where {D, T}
    return interpolate_quadratic_quadrilateral(q.vertices, r)
end

# -- IO --

function Base.show(io::IO, q::QuadraticQuadrilateral{D, T}) where {D, T}
    print(io, "QuadraticQuadrilateral", D) 
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
               q.vertices[4], ", ",
               q.vertices[5], ", ",
               q.vertices[6], ", ",
               q.vertices[7], ", ",
               q.vertices[8], ")")
end
