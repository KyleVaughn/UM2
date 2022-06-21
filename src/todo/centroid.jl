centroid(quad8::QuadraticQuadrilateral2D) = centroid(quad8, Val(3))
function centroid(quad8::QuadraticQuadrilateral{Dim, T}, ::Val{N}) where {Dim, T, N}
    # Gauss-Legendre quadrature over a quadrilateral is used.
    # Let Q(r,s) be the interpolation function for quad8,
    #            1  1                        N   N               
    # A = ∬ dA = ∫  ∫ ‖∂Q/∂r × ∂Q/∂s‖ds dr = ∑   ∑ wᵢwⱼ‖∂Q/∂r(rᵢ,sⱼ) × ∂Q/∂s(rᵢ,sⱼ)‖
    #     S      0  0                       j=1 i=1
    #                  1  N   N               
    # 𝗖 = (∬ 𝘅 dA)/A = -  ∑   ∑ 𝘅ᵢⱼwᵢwⱼ‖∂Q/∂r(rᵢ,sⱼ) × ∂Q/∂s(rᵢ,sⱼ)‖
    #      S           A j=1 i=1
    w, r = gauss_legendre_quadrature(T, Val(N))
    A = zero(T)
    𝗖 = @SVector zeros(T, Dim)
    for j ∈ 1:N, i ∈ 1:N
        J = 𝗝(quad8, r[i], r[j])
        weighted_val = w[i]*w[j]*norm(view(J, :, 1) × view(J, :, 2))
        𝗖 += weighted_val * quad8(r[i], r[j]).coord
        A += weighted_val
    end
    return Point(𝗖)/A
end

centroid(tri6::QuadraticTriangle2D) = centroid(tri6, Val(6))
function centroid(tri6::QuadraticTriangle{Dim, T}, ::Val{N}) where {Dim, T, N} 
    # Gauss-Legendre quadrature over a triangle is used.
    # Let F(r,s) be the interpolation function for tri6,
    #            1 1-r                       N                
    # A = ∬ dA = ∫  ∫ ‖∂F/∂r × ∂F/∂s‖ds dr = ∑ wᵢ‖∂F/∂r(rᵢ,sᵢ) × ∂F/∂s(rᵢ,sᵢ)‖
    #     S      0  0                       i=1
    #                  1  N                                 
    # 𝗖 = (∬ 𝘅 dA)/A = -  ∑ 𝘅 wᵢ‖∂F/∂r(rᵢ,sᵢ) × ∂F/∂s(rᵢ,sᵢ)‖ 
    #      S           A i=1
    w, r, s = gauss_legendre_quadrature(tri6, Val(N))
    A = zero(T)
    𝗖 = @SVector zeros(T, Dim)
    for i ∈ 1:N
        J = 𝗝(tri6, r[i], s[i])
        weighted_val = w[i] * norm(view(J, :, 1) × view(J, :, 2)) 
        𝗖 += weighted_val * tri6(r[i], s[i])
        A += weighted_val
    end
    return Point(𝗖)/A
end
