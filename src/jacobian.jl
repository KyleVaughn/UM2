jacobian(q::QuadraticSegment, r) = (4r - 3)*(q.𝘅₁ - q.𝘅₃) + (4r - 1)*(q.𝘅₂ - q.𝘅₃) 

function jacobian(quad::Quadrilateral, r, s)
    ∂F_∂r = (1 - s)*(quad[2] - quad[1]) + s*(quad[3] - quad[4])
    ∂F_∂s = (1 - r)*(quad[4] - quad[1]) + r*(quad[3] - quad[2])
    return hcat(∂F_∂r, ∂F_∂s)
end

function jacobian(tri6::QuadraticTriangle, r, s)
    ∂F_∂r = (4r + 4s - 3)tri6[1] +
                 (4r - 1)tri6[2] +
          (4(1 - 2r - s))tri6[4] +
                     (4s)tri6[5] +
                    (-4s)tri6[6]

    ∂F_∂s = (4r + 4s - 3)tri6[1] +
                 (4s - 1)tri6[3] +
                    (-4r)tri6[4] +
                     (4r)tri6[5] +
          (4(1 - r - 2s))tri6[6]
    return hcat(∂F_∂r, ∂F_∂s)
end

function jacobian(quad8::QuadraticQuadrilateral, r, s)
    # Chain rule
    # ∂F   ∂F ∂ξ     ∂F      ∂F   ∂F ∂η     ∂F
    # -- = -- -- = 2 -- ,    -- = -- -- = 2 --
    # ∂r   ∂ξ ∂r     ∂ξ      ∂s   ∂η ∂s     ∂η
    ξ = 2r - 1; η = 2s - 1
    ∂F_∂ξ = ((1 - η)*(2ξ + η)/4)quad8[1] +
            ((1 - η)*(2ξ - η)/4)quad8[2] +
            ((1 + η)*(2ξ + η)/4)quad8[3] +
            ((1 + η)*(2ξ - η)/4)quad8[4] +
                    (-ξ*(1 - η))quad8[5] +
                   ((1 - η^2)/2)quad8[6] +
                    (-ξ*(1 + η))quad8[7] +
                  (-(1 - η^2)/2)quad8[8]

    ∂F_∂η = ((1 - ξ)*( ξ + 2η)/4)quad8[1] +
            ((1 + ξ)*(-ξ + 2η)/4)quad8[2] +
            ((1 + ξ)*( ξ + 2η)/4)quad8[3] +
            ((1 - ξ)*(-ξ + 2η)/4)quad8[4] +
                   (-(1 - ξ^2)/2)quad8[5] +
                     (-η*(1 + ξ))quad8[6] +
                    ((1 - ξ^2)/2)quad8[7] +
                     (-η*(1 - ξ))quad8[8]

    return 2*hcat(∂F_∂ξ, ∂F_∂η)
end
