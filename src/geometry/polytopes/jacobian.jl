export jacobian

# 1-polytopes
jacobian(l::LineSegment, r) = l[2] - l[1]
jacobian(q::QuadraticSegment, r) = (4r - 3)*(q[1] - q[3]) + (4r - 1)*(q[2] - q[3]) 

# 2-polytopes
function jacobian(tri::Triangle, r, s)
    ∂F_∂r = tri[2] - tri[1]
    ∂F_∂s = tri[3] - tri[1]
    return hcat(∂F_∂r, ∂F_∂s)
end

function jacobian(quad::Quadrilateral, r, s)
    ∂F_∂r = (1 - s)*(quad[2] - quad[1]) + s*(quad[3] - quad[4])
    ∂F_∂s = (1 - r)*(quad[4] - quad[1]) + r*(quad[3] - quad[2])
    return hcat(∂F_∂r, ∂F_∂s)
end

function jacobian(tri6::QuadraticTriangle, r, s)
    ∂F_∂r = (4r + 4s - 3)*(tri6[1] - tri6[4]) +
                 (4r - 1)*(tri6[2] - tri6[4]) +
                     (4s)*(tri6[5] - tri6[6])

    ∂F_∂s = (4r + 4s - 3)*(tri6[1] - tri6[6]) +
                 (4s - 1)*(tri6[3] - tri6[6]) +
                     (4r)*(tri6[5] - tri6[4])
    return hcat(∂F_∂r, ∂F_∂s)
end

function jacobian(quad8::QuadraticQuadrilateral, r, s)
    # Chain rule
    # ∂F   ∂F ∂ξ     ∂F      ∂F   ∂F ∂η     ∂F
    # -- = -- -- = 2 -- ,    -- = -- -- = 2 --
    # ∂r   ∂ξ ∂r     ∂ξ      ∂s   ∂η ∂s     ∂η
    ξ = 2r - 1; η = 2s - 1
    ∂F_∂ξ = (η*(1 - η)/2)*(quad8[1] - quad8[2]) + 
            (η*(1 + η)/2)*(quad8[3] - quad8[4]) +
            (ξ*(1 - η)  )*(
                          (quad8[1] - quad8[5]) + 
                          (quad8[2] - quad8[5])
                          ) +
            (ξ*(1 + η)  )*(
                          (quad8[3] - quad8[7]) +
                          (quad8[4] - quad8[7])
                          ) +
        ((1 + η)*(1 - η))*(quad8[6] - quad8[8])


    ∂F_∂η = (ξ*(1 - ξ)/2)*(quad8[1] - quad8[4]) + 
            (ξ*(1 + ξ)/2)*(quad8[3] - quad8[2]) +
            (η*(1 - ξ)  )*(
                          (quad8[1] - quad8[8]) + 
                          (quad8[4] - quad8[8])
                          ) +
            (η*(1 + ξ)  )*(
                          (quad8[3] - quad8[6]) +
                          (quad8[2] - quad8[6])
                          ) +
        ((1 + ξ)*(1 - ξ))*(quad8[7] - quad8[5])

    return hcat(∂F_∂ξ, ∂F_∂η)
end

# 3-polytopes
function jacobian(tet::Tetrahedron, r, s, t)
    ∂F_∂r = tet[2] - tet[1]
    ∂F_∂s = tet[3] - tet[1]
    ∂F_∂t = tet[4] - tet[1]
    return hcat(∂F_∂r, ∂F_∂s, ∂F_∂t)
end

function jacobian(hex::Hexahedron, r, s, t)
    ∂F_∂r = ((1 - s)*(1 - t))*(hex[2] - hex[1]) + 
            ((    s)*(1 - t))*(hex[3] - hex[4]) +
            ((1 - s)*(    t))*(hex[6] - hex[5]) +
            ((    s)*(    t))*(hex[7] - hex[8])

    ∂F_∂s = ((1 - r)*(1 - t))*(hex[4] - hex[1]) + 
            ((    r)*(1 - t))*(hex[3] - hex[2]) +
            ((1 - r)*(    t))*(hex[8] - hex[5]) +
            ((    r)*(    t))*(hex[7] - hex[6])

    ∂F_∂t = ((1 - r)*(1 - s))*(hex[5] - hex[1]) + 
            ((    r)*(1 - s))*(hex[6] - hex[2]) +
            ((1 - r)*(    s))*(hex[8] - hex[4]) +
            ((    r)*(    s))*(hex[7] - hex[3])
    return hcat(∂F_∂r, ∂F_∂s, ∂F_∂t)
end

function jacobian(tet::QuadraticTetrahedron, r, s, t)
    u = 1 - r - s - t
    ∂F_∂r =      (tet[ 1] - tet[2]) +
            (4u)*(tet[ 5] - tet[1]) +
            (4r)*(tet[ 2] - tet[5]) +
            (4s)*(tet[ 6] - tet[7]) +
            (4t)*(tet[ 9] - tet[8]) 

    ∂F_∂s =      (tet[ 1] - tet[3]) +
            (4u)*(tet[ 7] - tet[1]) +
            (4r)*(tet[ 6] - tet[5]) +
            (4s)*(tet[ 3] - tet[7]) +
            (4t)*(tet[10] - tet[8]) 

    ∂F_∂t =      (tet[ 1] - tet[4]) +
            (4u)*(tet[ 8] - tet[1]) +
            (4r)*(tet[ 9] - tet[5]) +
            (4s)*(tet[10] - tet[7]) +
            (4t)*(tet[ 4] - tet[8]) 
    return hcat(∂F_∂r, ∂F_∂s, ∂F_∂t)
end
