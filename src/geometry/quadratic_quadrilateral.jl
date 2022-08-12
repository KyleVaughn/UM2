export QuadraticQuadrilateral,
       QuadraticQuadrilateral2,
       QuadraticQuadrilateral2f,
       QuadraticQuadrilateral2d

export interpolate_quadratic_quadrilateral,
       jacobian_quadratic_quadrilateral,
       jacobian,
       area

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
                                             p5::T, p6::T, p7::T, p8::T, r, s) where {T}
    ξ = 2 * r - 1    
    η = 2 * s - 1    
    return ((1 - ξ) * (1 - η) * (-ξ - η - 1) / 4) * p1 +    
           ((1 + ξ) * (1 - η) * ( ξ - η - 1) / 4) * p2 +    
           ((1 + ξ) * (1 + η) * ( ξ + η - 1) / 4) * p3 +    
           ((1 - ξ) * (1 + η) * (-ξ + η - 1) / 4) * p4 +    
                      ((1 - ξ * ξ) * (1 - η) / 2) * p5 +    
                      ((1 - η * η) * (1 + ξ) / 2) * p6 +    
                      ((1 - ξ * ξ) * (1 + η) / 2) * p7 +    
                      ((1 - η * η) * (1 - ξ) / 2) * p8
end

function interpolate_quadratic_quadrilateral(vertices::Vec{8}, r, s)
    ξ = 2 * r - 1
    η = 2 * s - 1
    return ((1 - ξ) * (1 - η) * (-ξ - η - 1) / 4) * vertices[1] +
           ((1 + ξ) * (1 - η) * ( ξ - η - 1) / 4) * vertices[2] +
           ((1 + ξ) * (1 + η) * ( ξ + η - 1) / 4) * vertices[3] +
           ((1 - ξ) * (1 + η) * (-ξ + η - 1) / 4) * vertices[4] +
                      ((1 - ξ * ξ) * (1 - η) / 2) * vertices[5] +
                      ((1 - η * η) * (1 + ξ) / 2) * vertices[6] +
                      ((1 - ξ * ξ) * (1 + η) / 2) * vertices[7] +
                      ((1 - η * η) * (1 - ξ) / 2) * vertices[8]
end

function (q::QuadraticQuadrilateral{D, T})(r::T, s::T) where {D, T}
    return interpolate_quadratic_quadrilateral(q.vertices, r)
end

# -- Jacobian --

function jacobian_quadratic_quadrilateral(p1::T, p2::T, p3::T, p4::T, 
                                          p5::T, p6::T, p7::T, p8::T, r, s) where {T}
    ξ = 2 * r - 1
    η = 2 * s - 1
    ∂r = (η * (1 - η) / 2) * (p1 - p2) +
         (η * (1 + η) / 2) * (p3 - p4) +
         (ξ * (1 - η)    ) * (
                             (p1 - p5) +
                             (p2 - p5)
                             ) +
         (ξ * (1 + η)    ) * (
                             (p3 - p7) +
                             (p4 - p7)
                             ) +
         ((1 + η)*(1 - η)) * (p6 - p8)

    ∂s = (ξ * (1 - ξ) / 2) * (p1 - p4) +
         (ξ * (1 + ξ) / 2) * (p3 - p2) +
         (η * (1 - ξ)    ) * (
                             (p1 - p8) +
                             (p4 - p8)
                             ) +
         (η * (1 + ξ)    ) * (
                             (p3 - p6) +
                             (p2 - p6)
                             ) +
         ((1 + ξ)*(1 - ξ)) * (p7 - p5)

    return Mat(∂r, ∂s)
end

function jacobian_quadratic_quadrilateral(vertices::Vec{8}, r, s)
    ξ = 2 * r - 1
    η = 2 * s - 1
    ∂r = (η * (1 - η) / 2) * (vertices[1] - vertices[2]) +
         (η * (1 + η) / 2) * (vertices[3] - vertices[4]) +
         (ξ * (1 - η)    ) * (
                             (vertices[1] - vertices[5]) +
                             (vertices[2] - vertices[5])
                             ) +
         (ξ * (1 + η)    ) * (
                             (vertices[3] - vertices[7]) +
                             (vertices[4] - vertices[7])
                             ) +
         ((1 + η)*(1 - η)) * (vertices[6] - vertices[8])

    ∂s = (ξ * (1 - ξ) / 2) * (vertices[1] - vertices[4]) +
         (ξ * (1 + ξ) / 2) * (vertices[3] - vertices[2]) +
         (η * (1 - ξ)    ) * (
                             (vertices[1] - vertices[8]) +
                             (vertices[4] - vertices[8])
                             ) +
         (η * (1 + ξ)    ) * (
                             (vertices[3] - vertices[6]) +
                             (vertices[2] - vertices[6])
                             ) +
         ((1 + ξ)*(1 - ξ)) * (vertices[7] - vertices[5])

    return Mat(∂r, ∂s)
end

function jacobian(q::QuadraticQuadrilateral{D, T}, r::T, s::T) where {D, T}
    return jacobian_quadratic_quadrilateral(q.vertices, r, s)
end

# -- Measure --

function area(q::QuadraticQuadrilateral{2, T}) where {T}
    # The area enclosed by the 4 quadratic edges + the area enclosed
    # by the quadrilateral (p1, p2, p3, p4)
    edge_area = T(2//3) * ((q[5] - q[1]) × (q[2] - q[1])  +
                           (q[6] - q[2]) × (q[3] - q[2])  +
                           (q[7] - q[3]) × (q[4] - q[3])  +
                           (q[8] - q[4]) × (q[1] - q[4]))

    quad_area = ((q[2] - q[1]) × (q[3] - q[1]) -       
                 (q[4] - q[1]) × (q[3] - q[1]))  / 2

    return edge_area + quad_area
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
