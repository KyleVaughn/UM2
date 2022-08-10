export interpolate_line_segment,
       interpolate_quadratic_segment,
       interpolate_triangle,
       interpolate_quadrilateral,
       interpolate_quadratic_triangle,
       interpolate_quadratic_quadrilateral

# -- Interpolation --

# See The Visualization Toolkit: An Object-Oriented Approach to 3D Graphics, 
# 4th Edition, Chapter 8, Advanced Data Representation

# Turn off the JuliaFormatter
#! format: off

# 1-polytope
function interpolate_line_segment(p1::T, p2::T, r) where {T} 
    return p1 + r * (p2 - p1)
end

function interpolate_line_segment(vertices::Vec{2, T}, r) where {T}
    return vertices[1] + r * (vertices[2] - vertices[1])
end

function interpolate_quadratic_segment(p1::T, p2::T, p3::T, r) where {T}
    return ((2 * r - 1) * (r - 1)) * p1 +    
           ((2 * r - 1) *  r     ) * p2 +    
           (-4 * r      * (r - 1)) * p3
end

function interpolate_quadratic_segment(vertices::Vec{3, T}, r) where {T}
    return ((2 * r - 1) * (r - 1)) * vertices[1] +    
           ((2 * r - 1) *  r     ) * vertices[2] +    
           (-4 * r      * (r - 1)) * vertices[3]
end

# 2-polytope
function interpolate_triangle(p1::T, p2::T, p3::T, r, s) where {T}
    return (1 - r - s) * p[1] + r * p[2] + s * p[3]
end

function interpolate_triangle(vertices::Vec{3, T}, r, s) where {T}
    return (1 - r - s) * vertices[1] + r * vertices[2] + s * vertices[3]
end

function interpolate_quadrilateral(p1::T, p2::T, p3::T, p4::T, r, s) where {T}
    return ((1 - r) * (1 - s)) * p1 +    
           (     r  * (1 - s)) * p2 +    
           (     r  *      s ) * p3 +    
           ((1 - r) *      s ) * p4
end

function interpolate_quadrilateral(vertices::Vec{4, T}, r, s) where {T}
    return ((1 - r) * (1 - s)) * vertices[1] +    
           (     r  * (1 - s)) * vertices[2] +    
           (     r  *      s ) * vertices[3] +    
           ((1 - r) *      s ) * vertices[4]
end

function interpolate_quadratic_triangle(p1::T, p2::T, p3::T, p4::T, p5::T, p6::T, r, s) where {T}
    return ((2 * (1 - r - s) - 1) * ( 1 - r - s)) * p1 +    
           (          r           * ( 2 * r - 1)) * p2 +    
           (              s       * ( 2 * s - 1)) * p3 +    
           (      4 * r           * ( 1 - r - s)) * p4 +    
           (      4 * r           *           s ) * p5 +    
           (          4 * s       * ( 1 - r - s)) * p6 
end

function interpolate_quadratic_triangle(vertices::Vec{6, T}, r, s) where {T}
    return ((2 * (1 - r - s) - 1) * ( 1 - r - s)) * vertices[1] +    
           (          r           * ( 2 * r - 1)) * vertices[2] +    
           (              s       * ( 2 * s - 1)) * vertices[3] +    
           (      4 * r           * ( 1 - r - s)) * vertices[4] +    
           (      4 * r           *           s ) * vertices[5] +    
           (          4 * s       * ( 1 - r - s)) * vertices[6]
end

function interpolate_quadratic_quadrilateral(p1::T, p2::T, p3::T, p4::T, p5::T, p6::T, p7::T, p8::T, r, s) where {T}
    ξ = 2r - 1; η = 2s - 1
    return ((1 - ξ)*(1 - η)*(-ξ - η - 1)/4) * p1 +  
           ((1 + ξ)*(1 - η)*( ξ - η - 1)/4) * p2 + 
           ((1 + ξ)*(1 + η)*( ξ + η - 1)/4) * p3 + 
           ((1 - ξ)*(1 + η)*(-ξ + η - 1)/4) * p4 + 
                      ((1 - ξ^2)*(1 - η)/2) * p5 + 
                      ((1 - η^2)*(1 + ξ)/2) * p6 + 
                      ((1 - ξ^2)*(1 + η)/2) * p7 + 
                      ((1 - η^2)*(1 - ξ)/2) * p8 
end

function interpolate_quadratic_quadrilateral(vertices::Vec{8, T}, r, s) where {T}
    ξ = 2r - 1; η = 2s - 1
    return ((1 - ξ)*(1 - η)*(-ξ - η - 1)/4) * vertices[1] +  
           ((1 + ξ)*(1 - η)*( ξ - η - 1)/4) * vertices[2] + 
           ((1 + ξ)*(1 + η)*( ξ + η - 1)/4) * vertices[3] + 
           ((1 - ξ)*(1 + η)*(-ξ + η - 1)/4) * vertices[4] + 
                      ((1 - ξ^2)*(1 - η)/2) * vertices[5] + 
                      ((1 - η^2)*(1 + ξ)/2) * vertices[6] + 
                      ((1 - ξ^2)*(1 + η)/2) * vertices[7] + 
                      ((1 - η^2)*(1 - ξ)/2) * vertices[8]
end

# # 3-polytope
# interpolation_weights(::Type{<:Tetrahedron}, r, s, t) = Vec((1 - r - s - t), r, s, t)
# interpolation_weights(::Type{<:Hexahedron},  r, s, t) = Vec((1 - r)*(1 - s)*(1 - t),
#                                                             (    r)*(1 - s)*(1 - t),
#                                                             (    r)*(    s)*(1 - t),
#                                                             (1 - r)*(    s)*(1 - t),
#                                                             (1 - r)*(1 - s)*(    t),
#                                                             (    r)*(1 - s)*(    t),
#                                                             (    r)*(    s)*(    t),
#                                                             (1 - r)*(    s)*(    t))
# function interpolation_weights(::Type{<:QuadraticTetrahedron}, r, s, t)
#     u = 1 - r - s - t
#     return Vec((2u-1)u,
#                (2r-1)r,
#                (2s-1)s,
#                (2t-1)t,
#                   4u*r,
#                   4r*s,
#                   4s*u, 
#                   4u*t,
#                   4r*t,
#                   4s*t)
# end
# function interpolation_weights(::Type{<:QuadraticHexahedron}, r, s, t)
#     ξ = 2r - 1; η = 2s - 1; ζ = 2t - 1
#     return Vec((1 - ξ)*(1 - η)*(1 - ζ)*(-2 - ξ - η - ζ)/8,
#                (1 + ξ)*(1 - η)*(1 - ζ)*(-2 + ξ - η - ζ)/8,
#                (1 + ξ)*(1 + η)*(1 - ζ)*(-2 + ξ + η - ζ)/8,
#                (1 - ξ)*(1 + η)*(1 - ζ)*(-2 - ξ + η - ζ)/8,
#                (1 - ξ)*(1 - η)*(1 + ζ)*(-2 - ξ - η + ζ)/8,
#                (1 + ξ)*(1 - η)*(1 + ζ)*(-2 + ξ - η + ζ)/8,
#                (1 + ξ)*(1 + η)*(1 + ζ)*(-2 + ξ + η + ζ)/8,
#                (1 - ξ)*(1 + η)*(1 + ζ)*(-2 - ξ + η + ζ)/8,
#                           (1 - ξ^2)*(1 - η  )*(1 - ζ  )/4,
#                           (1 + ξ  )*(1 - η^2)*(1 - ζ  )/4,
#                           (1 - ξ^2)*(1 + η  )*(1 - ζ  )/4,
#                           (1 - ξ  )*(1 - η^2)*(1 - ζ  )/4,
#                           (1 - ξ^2)*(1 - η  )*(1 + ζ  )/4,
#                           (1 + ξ  )*(1 - η^2)*(1 + ζ  )/4,
#                           (1 - ξ^2)*(1 + η  )*(1 + ζ  )/4,
#                           (1 - ξ  )*(1 - η^2)*(1 + ζ  )/4,
#                           (1 - ξ  )*(1 - η  )*(1 - ζ^2)/4,
#                           (1 + ξ  )*(1 - η  )*(1 - ζ^2)/4,
#                           (1 + ξ  )*(1 + η  )*(1 - ζ^2)/4,
#                           (1 - ξ  )*(1 + η  )*(1 - ζ^2)/4)
# end
