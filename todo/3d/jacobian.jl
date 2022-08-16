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
