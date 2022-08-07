
# Turn off the JuliaFormatter
#! format: off

# 1-polytopes
function jacobian_line_segment(p1::T, p2::T, r) where {T} 
    return p2 - p1
end

function jacobian_line_segment(vertices::Vec{2, T}, r) where {T} 
    return vertices[2] - vertices[1]
end

function jacobian_quadratic_segment(p1::T, p2::T, p3::T, r) where {T}
    return (4r - 3)*(p1 - p3) + 
           (4r - 1)*(p2 - p3) 
end

function jacobian_quadratic_segment(vertices::Vec{3, T}, r) where {T}
    return (4r - 3)*(vertices[1] - vertices[3]) + 
           (4r - 1)*(vertices[2] - vertices[3]) 
end

# 2-polytopes
function jacobian_triangle(p1::T, p2::T, p3::T, r, s) where {T}
    ∂ᵣF = p2 - p1
    ∂ₛF = p3 - p1
    return hcat(∂ᵣF, ∂ₛF)
end

function jacobian_triangle(vertices::Vec{3, T}, r, s) where {T}
    ∂ᵣF = vertices[2] - vertices[1]
    ∂ₛF = vertices[3] - vertices[1]
    return hcat(∂ᵣF, ∂ₛF)
end

function jacobian_quadrilateral(p1::T, p2::T, p3::T, p4::T, r, s) where {T}
    ∂ᵣF = (1 - s)*(p2 - p1) + s*(p3 - p4)
    ∂ₛF = (1 - r)*(p4 - p1) + r*(p3 - p2)
    return hcat(∂ᵣF, ∂ₛF)
end

function jacobian_quadrilateral(vertices::Vec{4, T}, r, s) where {T}
    ∂ᵣF = (1 - s)*(vertices[2] - vertices[1]) + s*(vertices[3] - vertices[4])
    ∂ₛF = (1 - r)*(vertices[4] - vertices[1]) + r*(vertices[3] - vertices[2])
    return hcat(∂ᵣF, ∂ₛF)
end

# function jacobian(tri6::QuadraticTriangle, r, s)
#     ∂ᵣF = (4r + 4s - 3)*(tri6[1] - tri6[4]) +
#                (4r - 1)*(tri6[2] - tri6[4]) +
#                    (4s)*(tri6[5] - tri6[6])
# 
#     ∂ₛF = (4r + 4s - 3)*(tri6[1] - tri6[6]) +
#                (4s - 1)*(tri6[3] - tri6[6]) +
#                    (4r)*(tri6[5] - tri6[4])
#     return hcat(∂ᵣF, ∂ₛF)
# end
# 
# function jacobian(quad8::QuadraticQuadrilateral, r, s)
#     # Chain rule
#     # ∂F   ∂F ∂ξ     ∂F      ∂F   ∂F ∂η     ∂F
#     # -- = -- -- = 2 -- ,    -- = -- -- = 2 --
#     # ∂r   ∂ξ ∂r     ∂ξ      ∂s   ∂η ∂s     ∂η
#     ξ = 2r - 1; η = 2s - 1
#     ∂F_∂ξ = (η*(1 - η)/2)*(quad8[1] - quad8[2]) + 
#             (η*(1 + η)/2)*(quad8[3] - quad8[4]) +
#             (ξ*(1 - η)  )*(
#                           (quad8[1] - quad8[5]) + 
#                           (quad8[2] - quad8[5])
#                           ) +
#             (ξ*(1 + η)  )*(
#                           (quad8[3] - quad8[7]) +
#                           (quad8[4] - quad8[7])
#                           ) +
#         ((1 + η)*(1 - η))*(quad8[6] - quad8[8])
# 
# 
#     ∂F_∂η = (ξ*(1 - ξ)/2)*(quad8[1] - quad8[4]) + 
#             (ξ*(1 + ξ)/2)*(quad8[3] - quad8[2]) +
#             (η*(1 - ξ)  )*(
#                           (quad8[1] - quad8[8]) + 
#                           (quad8[4] - quad8[8])
#                           ) +
#             (η*(1 + ξ)  )*(
#                           (quad8[3] - quad8[6]) +
#                           (quad8[2] - quad8[6])
#                           ) +
#         ((1 + ξ)*(1 - ξ))*(quad8[7] - quad8[5])
# 
#     return hcat(∂F_∂ξ, ∂F_∂η)
# end
# 
# # 3-polytopes
# function jacobian(tet::Tetrahedron, r, s, t)
#     ∂ᵣF = tet[2] - tet[1]
#     ∂ₛF = tet[3] - tet[1]
#     ∂ₜF = tet[4] - tet[1]
#     return hcat(∂ᵣF, ∂ₛF, ∂ₜF)
# end
# 
# function jacobian(hex::Hexahedron, r, s, t)
#     ∂ᵣF = ((1 - s)*(1 - t))*(hex[2] - hex[1]) + 
#           ((    s)*(1 - t))*(hex[3] - hex[4]) +
#           ((1 - s)*(    t))*(hex[6] - hex[5]) +
#           ((    s)*(    t))*(hex[7] - hex[8])
# 
#     ∂ₛF = ((1 - r)*(1 - t))*(hex[4] - hex[1]) + 
#           ((    r)*(1 - t))*(hex[3] - hex[2]) +
#           ((1 - r)*(    t))*(hex[8] - hex[5]) +
#           ((    r)*(    t))*(hex[7] - hex[6])
# 
#     ∂ₜF = ((1 - r)*(1 - s))*(hex[5] - hex[1]) + 
#           ((    r)*(1 - s))*(hex[6] - hex[2]) +
#           ((1 - r)*(    s))*(hex[8] - hex[4]) +
#           ((    r)*(    s))*(hex[7] - hex[3])
#     return hcat(∂ᵣF, ∂ₛF, ∂ₜF)
# end
# 
# function jacobian(tet::QuadraticTetrahedron, r, s, t)
#     u = 1 - r - s - t
#     ∂ᵣF =      (tet[ 1] - tet[2]) +
#           (4u)*(tet[ 5] - tet[1]) +
#           (4r)*(tet[ 2] - tet[5]) +
#           (4s)*(tet[ 6] - tet[7]) +
#           (4t)*(tet[ 9] - tet[8]) 
# 
#     ∂ₛF =      (tet[ 1] - tet[3]) +
#           (4u)*(tet[ 7] - tet[1]) +
#           (4r)*(tet[ 6] - tet[5]) +
#           (4s)*(tet[ 3] - tet[7]) +
#           (4t)*(tet[10] - tet[8]) 
# 
#     ∂ₜF =      (tet[ 1] - tet[4]) +
#           (4u)*(tet[ 8] - tet[1]) +
#           (4r)*(tet[ 9] - tet[5]) +
#           (4s)*(tet[10] - tet[7]) +
#           (4t)*(tet[ 4] - tet[8]) 
#     return hcat(∂ᵣF, ∂ₛF, ∂ₜF)
# end
